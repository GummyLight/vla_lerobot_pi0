# 自适应 IK 改进方案（Adaptive IK Improvements）

## 问题背景

在 `base_biased_servoj` 模式下，机械臂在以下情况会进入保护状态：
1. **底座摇摆**：小范围TCP运动 → 底座无条件跟随方位角变化 → 400°+的总摇摆
2. **工作空间压缩**：底座转到 ±90° 以上 → 腕部工作空间极度压缩 → IK无解
3. **自碰撞**：底座与机械臂本体自碰撞

## 解决方案：`base_biased_adaptive` 模式

### 核心改进（三层防护）

```
IK流程：
  1. 计算底座偏置需求
       ↓
  2. [新] 自适应阻尼 - 接近±150°时减弱偏置
       ↓
  3. [新] EMA平滑 - 过滤方位角噪声
       ↓
  4. 调用 ur_rtde IK求解
       ↓
  5. [新] 如果失败 → 回退到当前关节角作为seed，重试一次
```

### 改进细节

#### 1. **自适应阻尼**（Adaptive Damping）

```python
# 当底座角度接近极限时，逐渐减弱偏置需求
proximity = abs(q[0]) / base_limit_rad  # 0 ~ 1.0
if proximity > damping_threshold (0.8):  # 80% 处开始
    damping = max(0.2, 1.5 - proximity)  # 线性衰减到20%
    bias_raw *= damping
```

**效果**：
- 底座在 ±2.6 rad (±150°) 之前正常工作
- 接近极限时自动减弱，避免进入自碰撞区
- 完全不动的"死区"来自于工作空间物理约束，不是算法限制

#### 2. **EMA 平滑**（Exponential Moving Average）

```python
bias_smooth = 0.3 * bias_raw + 0.7 * last_bias
```

**效果**：
- 方位角噪声（±几度）不再直接传到底座
- 小范围抖动时底座基本不动
- 大范围运动时平顺跟随

#### 3. **IK 失败回退**（Fallback）

```python
q_target = get_inverse_kinematics(pose_cmd, q_seed_biased)
if q_target is None:
    # 回退到保守策略
    q_target = get_inverse_kinematics(pose_cmd, q_current)
```

**效果**：
- 当 biased seed 导致IK失败时，自动回退
- 从"不可达"错误 → 能执行（可能用wrist代偿）

---

## 配置参数

在 `pika_config.yaml` 中：

```yaml
teleoperation:
  # 启用新的自适应模式（推荐）
  ik_mode: base_biased_adaptive
  
  # 或保持原来的模式
  # ik_mode: base_biased_servoj
  # ik_mode: ur_native_servol
  
  # TCP离底座足够远时启用底座偏置 (m)
  base_bias_min_radius_m: 0.15  # 推荐值 (之前: 0.05)
  
  # 自适应阻尼参数 (adaptive模式专用)
  base_limit_rad: 2.6              # 底座旋转极限 (rad, ≈150°)
  base_limit_damping_threshold: 0.8  # 80%处开始阻尼
```

### 参数调优指南

| 参数 | 范围 | 推荐 | 调大 ↔ 调小 |
|-----|-----|------|-----------|
| `base_limit_rad` | 1.5~3.0 | 2.6 rad | 调大允许更多底座旋转；调小保守 |
| `base_bias_min_radius_m` | 0.05~0.3 | 0.15 | 调大减少摇摆；调小允许近轴偏置 |
| `base_limit_damping_threshold` | 0.6~0.95 | 0.8 | 调小更早开始减弱；调大允许更大旋转 |

---

## 三种 IK 模式对比

### 方案 A：`ur_native_servol`（UR内置IK）

**原理**：
- 每帧发送目标TCP给UR控制器
- UR自动选择最接近当前Q的IK分支
- 使用 servoL（TCP速度控制）

**优点**：
- 平顺，无跳跃
- 不需要预计算

**缺点**：
- 大范围圆形运动时，UR常选择"腕部体操"方案而不是旋转底座
- 腕部关节可能快速达到极限

### 方案 B：`base_biased_servoj`（带偏置的IK求解）

**原理**：
- 每帧计算目标XY方位角 → 底座应指向角度
- 将底座seed偏移到该角度
- ur_rtde 求解IK → 用 servoJ（关节速度控制）

**优点**：
- 底座主动做圆形运动，腕部负荷低
- 工作空间利用率高

**缺点**：
- 底座可能无条件跟随方位角变化
- 接近极限时易进自碰撞

### 方案 C：`base_biased_adaptive`（推荐 ⭐）

**改进**：
- 方案B + 自适应阻尼 + EMA平滑 + IK失败回退
- 三层防护自动避免保护状态

**适用**：
- 一般遥操作（最佳选择）
- 需要大范围运动但要避免自碰撞
- 方位角噪声大的跟踪系统

---

## 测试清单

执行遥操作测试时注意观察：

- [ ] **底座运动**
  - 小范围TCP抖动时，底座是否基本不动？ (EMA效果)
  - 大范围圆形运动时，底座是否平顺旋转？ (偏置效果)

- [ ] **接近极限**
  - 将arm手动转到 q[0] ≈ ±120° (接近 2.6 rad 的46%)
  - 用遥操作器在远处绕圆形运动
  - 底座是否自动减弱运动？ (阻尼效果)

- [ ] **保护状态**
  - 记录遥操作 5~10 分钟
  - 是否仍然发生"不可达"保护停止？ (回退效果)

- [ ] **工作空间**
  - 尝试TCP指向底座（距离 < 0.15m）的方向
  - 手臂是否能平稳执行（不跳到远处）？

---

## 实现细节（代码）

### collect_pika.py 中的改动

在主IK循环（第~592行）：

```python
if self.ik_mode in ("base_biased_servoj", "base_biased_adaptive"):
    # 计算底座偏置
    d_raw = (tgt_az - cur_az + π) % (2π) - π
    
    if self.ik_mode == "base_biased_adaptive":
        # 1. 自适应阻尼
        proximity = abs(q_current[0]) / self.base_limit_rad
        if proximity > self.base_limit_damping_threshold:
            damping = max(0.2, 1.5 - proximity)
            d_raw *= damping
        
        # 2. EMA 平滑
        d = 0.3 * d_raw + 0.7 * self._last_base_bias
        self._last_base_bias = d
    else:
        d = d_raw
    
    q_seed[0] = q_current[0] + d
    q_target = self.robot.get_inverse_kinematics(pose_cmd, q_seed)
    
    # 3. IK 失败回退
    if q_target is None and self.ik_mode == "base_biased_adaptive":
        q_target = self.robot.get_inverse_kinematics(pose_cmd, q_current)
```

### 参数传递链

```
pika_config.yaml 
  ↓ [yaml loads]
teleop_only.py / collect_pika.py
  ↓ [reads cfg]
PikaTeleopController.__init__
  ↓ [receives base_limit_rad, base_limit_damping_threshold]
_loop() method
  ↓ [uses for adaptive damping]
safe IK + servo commands
```

---

## 故障排查

| 症状 | 原因 | 解决方案 |
|------|------|--------|
| 保护状态仍频发 | IK失败回退还不够 | 增加 `base_bias_min_radius_m` (远离底座轴) |
| 底座运动太慢 | 阻尼过强 | 减小 `base_limit_damping_threshold` (更晚开始衰减) |
| 底座运动太快 | 阻尼太弱 | 增加 `base_limit_damping_threshold` 或减小 `base_limit_rad` |
| 方位角噪声导致摇摆 | EMA平滑不足 | 改hardcoded的EMA系数 0.3→0.2 (更多历史权重) |
| 自碰撞 | 底座转过极限 | 减小 `base_limit_rad` 到 2.0~2.2 |

---

## 进阶：下一步优化

### 优先级

1. **⭐ 推荐**：收集 joint_limits 边界 (使用 `tools/probe_joint_limits.py`)
   - 手动freedrive学习自碰撞边界
   - 在 `pika_config.yaml` 中设置每个关节的安全限
   - 最终解决方案：工作空间级预防

2. **可选**：混合IK模式
   - 小范围精准操作 → `ur_native_servol`
   - 大范围运动 → `base_biased_adaptive`
   - 自动切换基于TCP运动幅度

3. **高级**：路径规划感知IK
   - 预检查目标是否会导致关节反向
   - 选择对关节友好的IK分支

---

## 参考

- [IK 模式详细对比](IK_MODES_COMPARISON.md)
- [UR7e 底座偏置陷阱分析](ik_bias_demo.py)
- UR RealTime Data Exchange (RTDE) 文档
