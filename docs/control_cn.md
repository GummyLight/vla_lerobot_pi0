# 控制流程指南(中文)

> English: [docs/control.md](control.md)

本文说明 [scripts/run_pi0_robot.py](../scripts/run_pi0_robot.py) 在真机上是怎么从「策略输出」一路走到「机械臂动 + 夹爪开合」的。理解这个之后,调参 / 排错才有方向。

## 一图看懂

```
┌──────────────────────┐    state(7) + 2张RGB图    ┌──────────────────┐
│  RealSense × 2       │ ───────────────────────▶ │                  │
│  + UR getActualQ()   │                          │   pi0 + LoRA     │ action chunk
│  + Robotiq GET POS   │                          │   policy         │ (50 × 7)
└──────────────────────┘                          └──────────────────┘
                                                           │
                            ┌──────────────────────────────┤ 每步 pop 一帧 (7,)
                            │                              │
                            ▼                              ▼
              ┌───────────────────────────┐
              │ 安全闸刀                  │
              │ |Δq|.max() < 0.10 rad?    │
              │ 超过 → 保持上一帧         │
              └─────────┬─────────────────┘
                        │ q[:6] 关节角            │ q[6] 夹爪
                        ▼                          ▼
              ┌─────────────────────┐   ┌──────────────────────┐
              │ rtde_control        │   │ socket :63352        │
              │   .servoJ(...)      │   │   "SET POS 0..255\n" │
              │   30Hz, gain=300    │   │                      │
              └────────┬────────────┘   └─────────┬────────────┘
                       │ RTDE @ 500Hz             │ TCP / ASCII
                       ▼                          ▼
            ┌──────────────────────┐   ┌──────────────────────────┐
            │ UR 控制器            │   │ Robotiq URCap → 驱动板   │
            │ 内部插值伺服         │   │                          │
            └────────┬─────────────┘   └─────────┬────────────────┘
                     │                            │
                     ▼                            ▼
              6 个关节伺服电机              夹爪夹紧/张开
```

---

## 阶段 1 — 策略推理(pi0 + chunk 队列)

[scripts/run_pi0_robot.py:404-409](../scripts/run_pi0_robot.py#L404-L409)

```python
with torch.no_grad():
    action = policy.select_action(obs)   # 7 维 numpy
target_joints = action[:6]               # 6 个关节角(弧度,绝对位置)
target_gripper = float(action[6])        # 夹爪 [0, 1]
```

pi0 不是「每个控制周期都重新跑一次大模型」。它在内部维护一个**长度 50 的动作队列**(`chunk_size=50`,30 Hz 数据下 ≈ 1.66 秒):

- **队列非空**:`select_action` 直接 pop 一帧返回(几毫秒,纯 GPU↔CPU 数据搬运)。
- **队列空**:重跑一次扩散前向,生成下一段 50 帧的 chunk(实测 ~430 ms)。

这是 pi0 在 30 Hz 闭环里能跑得动的关键。代价:每 50 步会有一次 ~430 ms 的尖峰延迟(就是你 log 里 `step 51 / 101 / 151... control loop slow` 的来源),无法在不改架构的前提下消除。

理论上稳态:`50 步 × 33 ms + 1 chunk × 430 ms = 2080 ms / 50 步 = 24 Hz` 是上限。要冲到稳定 30 Hz 必须做异步 chunk 生成(后台线程预生成下一批,主循环只 pop)。

---

## 阶段 2 — 安全闸刀

[scripts/run_pi0_robot.py:412-417](../scripts/run_pi0_robot.py#L412-L417)

```python
if prev_target is not None:
    delta = np.abs(target_joints - prev_target).max()
    if delta > args.max_joint_delta_rad:    # 默认 0.10 rad
        print("⚠ refusing to send.")
        target_joints = prev_target          # 冻结在上一帧
```

- 阈值 `0.10 rad ≈ 5.7°`,在 30 Hz 下等效角速度 **3 rad/s ≈ 172°/s**。这是「正常 demo 速度的上限」附近。
- 触发条件:模型预测的下一步目标和**上一帧实际下发的**关节角差超过阈值。
- 触发后:不下发新目标,把 `target_joints` 改回 `prev_target`,机械臂保持静止。
- 注意:**pi0 的 chunk 队列仍然在推进**——`select_action` 每次都会 pop 一帧,只是这一帧被我们 ignored。所以「安全冻结期间」chunk 还在往前消耗。

什么时候应该调宽?
- 模型已经训得不错、动作本身就是快速运动(比如急速避障)→ 可调到 `0.15 ~ 0.20`。
- **欠训模型 → 不要放宽**,放宽就是允许机械臂乱舞。

什么时候应该调严?
- 第一次试机、不信任模型 → `0.05`。
- 动 EE 上有重物或脆弱工件 → `0.05`。

---

## 阶段 3 — 关节控制(servoJ + RTDE)

[scripts/run_pi0_robot.py:328-339](../scripts/run_pi0_robot.py#L328-L339)

```python
def send_joint_target(self, q, dt=None):
    sj = self.servoj
    t = dt if (dt is not None and dt > 0) else (1.0 / self.control_hz)
    self.ctrl.servoJ(q.tolist(),
                     sj["velocity"], sj["acceleration"],
                     t,                       # 关键!
                     sj["lookahead_time"], sj["gain"])
```

走的是 UR 官方 **`servoJ`**——专门给「在线轨迹流式喂点」设计的实时伺服模式。底层走 **RTDE 协议(Real-Time Data Exchange,500 Hz)**,由 `ur_rtde` 这个 C++/Python 库经 `rtde_control.RTDEControlInterface` 包装。

参数含义:

| 参数 | 含义 | 推荐起点 | 调整影响 |
|---|---|---|---|
| `q` | 6 个目标关节角(弧度) | — | 模型输出,不调 |
| `velocity` | 控制器内部速度上限(rad/s) | 0.5 | 大→更激进,小→更保守 |
| `acceleration` | 控制器内部加速度上限(rad/s²) | 0.5 | 同上 |
| **`time`** | **告诉控制器「`time` 秒后到达 q」** | **实测周期 dt** | **失配会颠簸,见下** |
| `lookahead_time` | 控制器内部预瞄时长(s) | 0.10 | 大→更平滑但跟随滞后,小→更跟手但易抖(0.03-0.20 区间) |
| `gain` | 闭环刚度 | 300 | 大→响应快但易振荡(100-600 区间) |

### `time` 必须等于实测周期(关键)

`time` 是「控制器在多长时间内插值到目标位置」。如果你**循环实际 80 ms 才喂一次**点,但 `time=33ms`,控制器会:

- t=0: 收到指令 A,33 ms 内**全速冲**到 A → 然后干站 47 ms
- t=80: 收到指令 B,33 ms 内**全速冲**到 B → 然后干站 47 ms
- ...

→ **急走-急停的颠簸**,而不是平滑伺服。

我们的实现 [scripts/run_pi0_robot.py:467](../scripts/run_pi0_robot.py#L467) 把每一轮的实际墙钟周期 `prev_loop_dt` 喂给下一轮的 `servoJ time`:

```python
prev_loop_dt = max(time.perf_counter() - tick, period)
```

这样不管循环跑多慢,运动都是**连续匀速**(只是整体放慢到 0.8x / 0.5x...),不会颠簸。

### RTDE 的两个 Interface

- `RTDEControlInterface(ip)` → 发指令(servoJ / movel / forceMode 等),`servoStop()` / `stopScript()` 退出
- `RTDEReceiveInterface(ip)` → 读状态(`getActualQ()` / `getActualTCPPose()` 等)

这俩是独立连接,共享 UR 的 RTDE 端口 30004。

---

## 阶段 4 — 夹爪控制(Robotiq URCap socket)

[scripts/run_pi0_robot.py:218-260](../scripts/run_pi0_robot.py#L218-L260)

夹爪**不走 RTDE**。Robotiq 的 URCap(PolyScope 上装的插件)在 UR 控制柜里**单独开了一个 TCP 端口 `63352`**,协议是非常简单的 ASCII 命令:

| 指令 | 含义 |
|---|---|
| `SET ACT 1\n` | 激活夹爪(第一次会做一个全开-全闭自检循环) |
| `SET GTO 1\n` | 启用 go-to-position 模式 |
| `SET SPE 200\n` | 速度(0-255) |
| `SET FOR 100\n` | 力(0-255) |
| `SET POS 0\n` | 全开 |
| `SET POS 255\n` | 全闭 |
| `GET POS\n` | 返回当前位置,`POS <0..255>\n` |

我们的封装 [scripts/run_pi0_robot.py:238-251](../scripts/run_pi0_robot.py#L238-L251):

```python
def write_position(self, value: float) -> None:
    """value ∈ [0, 1]: 0=open, 1=closed"""
    raw = int(round(max(0.0, min(1.0, value)) * 255))
    self._cmd(f"SET POS {raw}")
```

策略输出的 gripper 是 `[0, 1]` 浮点,我们映射到 `[0, 255]` 再发。

### 主循环里的二值化

[scripts/run_pi0_robot.py:421](../scripts/run_pi0_robot.py#L421)

```python
robot.send_gripper(1.0 if target_gripper > args.gripper_threshold else 0.0)
```

3D 打印机数据集的夹爪基本只有「全开 / 全闭」(开门 = 抓门把手),所以阈值化成二值更稳。如果你的任务有连续夹爪动作(比如夹软物),改成:

```python
robot.send_gripper(target_gripper)   # 直接连续值
```

### URCap socket 的「隐形」前置条件

**端口 63352 只在示教器上有 program 在 run 的时候才开放。** 这是 URCap 的设计 —— 工具栏代码作为 UR script 的一部分被嵌入到 program 里,不跑 program 就没人在监听这个端口。

第一次部署常见的坑:夹爪上电了、URCap 也装了,但没在 PolyScope 上跑一个含 Robotiq toolbar 的 program → `connection refused`。 [scripts/preflight_check.py](../scripts/preflight_check.py) 的第 3 项检查就是抓这个的。

---

## 控制循环全貌

[scripts/run_pi0_robot.py:382-468](../scripts/run_pi0_robot.py#L382-L468)

```python
period = 1.0 / args.control_hz   # 33.3 ms
prev_loop_dt = period            # 上一轮实测周期,初始用 period

while time.perf_counter() - t0 < max_seconds:
    tick = time.perf_counter()

    # 1. 观测
    joints = robot.get_joints()       # RTDE 读 6 个关节角
    gripper = robot.get_gripper()     # Robotiq socket 读位置
    img_g, img_w = cams.read()        # 两个 RealSense 并行读

    # 2. 推理
    obs = build_observation(joints, gripper, img_g, img_w, task)
    action = policy.select_action(pre(obs))   # pi0 chunk pop
    action = post(action).numpy()              # 反归一化 → (7,)

    # 3. 安全
    if abs(action[:6] - prev_target).max() > MAX_DELTA:
        action[:6] = prev_target              # 拒绝,冻结

    # 4. 下发
    robot.send_joint_target(action[:6], dt=prev_loop_dt)   # servoJ
    robot.send_gripper(action[6] > THRESHOLD)              # Robotiq SET POS

    prev_target = action[:6]

    # 5. 配速 + 记录本轮实际耗时
    elapsed = time.perf_counter() - tick
    if elapsed < period:
        time.sleep(period - elapsed)
    prev_loop_dt = max(time.perf_counter() - tick, period)
```

---

## 调参速查表(改 [configs/run_pi0_robot.yaml](../configs/run_pi0_robot.yaml))

| 现象 | 改哪个 | 怎么改 |
|---|---|---|
| 全程在拦截、机械臂不动 | `control.max_joint_delta_rad` | 先**别**调宽 → 大概率是模型欠训,接着训而不是放宽闸刀 |
| 机械臂动得很僵硬,「冲-停-冲」 | `robot.servoj.lookahead_time` | 0.10 → 0.15 |
| 机械臂高频抖动 | `robot.servoj.gain` | 300 → 200 |
| 机械臂跟不上目标(滞后) | `robot.servoj.gain` | 300 → 400 |
| 夹爪反应慢 | `RobotiqGripper._cmd("SET SPE ...")` | 200 → 255(代码里改) |
| 夹爪夹力不够 | `RobotiqGripper._cmd("SET FOR ...")` | 100 → 200 |
| 循环跑得慢(<20 Hz) | 优先看 chunk 边界(无解)/ 摄像头 / GPU | dry-run 看 `dt=` 数字 |
| 任务完全不会做 | **欠训** | `train_pi0.py --steps=30000 --resume=true` |

---

## 故障排查

| 症状 | 大概率原因 |
|---|---|
| `RTDEControlInterface` 卡住 / 报错 | 示教器不在 Remote Control 模式,或另一个 RTDE 客户端没断开 |
| `connection refused` to port 63352 | 示教器上没在跑含 Robotiq toolbar 的 program |
| `servoJ(): incompatible function arguments` | `ur_rtde` 版本对 servoJ 签名变了,我们已用位置参数 |
| 第一次跑夹爪开-关一下然后崩溃 | `SET ACT 1` 自检正常,崩溃看堆栈定位 |
| 100% 步触发 `max joint delta` | 模型欠训,见上文 |
| `control loop slow` 每 50 步一次 | pi0 chunk 边界,无解(除非异步化) |
| `control loop slow` 每步都有 | 摄像头 I/O 或 GPU 瓶颈,dry-run 看 `dt=` 量级 |
