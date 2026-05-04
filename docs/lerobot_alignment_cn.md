# 向 `huggingface/lerobot` 框架靠拢

> English version: [lerobot_alignment.md](lerobot_alignment.md).

本项目原本就紧贴上游 [`lerobot`](https://github.com/huggingface/lerobot)：训练
和数据集加载已经走 lerobot 的 CLI/数据集类，但**数据采集、真机推理、配置系统**
是围绕老的 `collect/` 工具包从零写的。本文档记录两套实现的差距，以及
`align-with-lerobot` 分支上做的对齐工作。

## TL;DR — 改动了什么

新建 `src/vla_pi0/` 包，目录布局完全对齐 lerobot：

```
src/vla_pi0/
├── robots/
│   └── ur7e_follower/                 # UR7e + Robotiq 2F-58 从动臂
│       ├── config_ur7e_follower.py    # @RobotConfig.register_subclass("ur7e_follower")
│       └── ur7e_follower.py           # UR7eFollower(Robot)
└── scripts/
    ├── record.py                      # 对应 lerobot-record
    └── rollout.py                     # 对应 lerobot-rollout
```

对齐之后，可以用上游 SO-100 / Koch 演示同样的方式驱动整套硬件：

```bash
# 闭环策略推理（legacy: deploy/run_pi0_robot.py）
python -m vla_pi0.scripts.rollout \
    --policy-path outputs/train/pi0_3d_printer_lora/checkpoints/last/pretrained_model \
    --task "open the 3D printer" \
    --robot.ip 192.168.1.100 \
    --robot.cameras.cam-global.serial <SN_A> \
    --robot.cameras.cam-wrist.serial  <SN_B>

# 演示数据采集（原来是 collect/collect_urscript.py / collect_pika.py）
python -m vla_pi0.scripts.record \
    --dataset-name my_demo \
    --task "open the 3D printer" \
    --source urscript \
    --urscript "collect/urscripts/open the 3D printer.script" \
    --robot.ip 192.168.1.100
```

老的 `collect/` 工具包和 `deploy/run_pi0_robot.py` 完全保留 —— 新包是**附加**的，
不替换。

## 逐项对比

| 关注点 | 上游 `lerobot` | 本项目（之前） | 本项目（现在） |
|---|---|---|---|
| Robot 抽象 | `lerobot.robots.robot.Robot` 抽象类，约定 `connect/disconnect/get_observation/send_action/observation_features/action_features/is_connected/is_calibrated` | 没有抽象 —— `UR7eInterface`、`RobotiqGripper`、`MultiCamera` 直接散在每个脚本里 | `vla_pi0.robots.ur7e_follower.UR7eFollower(Robot)` 完整实现契约 |
| Robot 配置 | `RobotConfig`（基于 `draccus.ChoiceRegistry`）+ `@RobotConfig.register_subclass("name")` | YAML 文件（`deploy/configs/run_pi0_robot.yaml`）每个脚本各自解析 | `UR7eFollowerConfig` 注册为 `ur7e_follower`；旧的 YAML 仍然能通过 `to_legacy_dict()` 兼容 |
| Camera 抽象 | `lerobot.cameras.realsense.RealSenseCamera`（注册名 `intelrealsense`） | `pyrealsense2` 的调用同时出现在 `collect/utils/camera_interface.py` 和 `deploy/run_pi0_robot.py` 两处 | 集中到 `UR7eFollower._open_cameras()` 一处 |
| 数据采集 | `lerobot-record` → `make_robot_from_config(...)` + `LeRobotDataset` writer | `collect/collect_urscript.py` + `collect/collect_pika.py`（自带 writer） | `vla_pi0.scripts.record` 通过 `Robot` 契约复用 `LeRobotWriter` |
| 真机推理 | `lerobot-rollout`（带 `strategy.type=base/sentry/...`） | `deploy/run_pi0_robot.py`（自己写的 Cameras + URRobot 类） | `vla_pi0.scripts.rollout`（走 Robot 抽象；strategy 风格的循环） |
| 训练 | `lerobot-train`（`TrainPipelineConfig`） | `train/train_pi0.py` 包装 `lerobot.scripts.lerobot_train` | 不变 —— 本来就对齐了 |
| 离线评估 | `lerobot-eval`（仅支持 gym vector env 仿真） | `eval/eval_pi0.py`（基于真实数据集的开环动作预测） | 不变 —— 这是 `lerobot-eval` 不覆盖的真实数据离线场景 |
| 数据集格式 | LeRobot v3.0（parquet + mp4，多 episode 分块） | LeRobot v3.0 | 不变 —— 已对齐 |
| CLI 配置 | `draccus`（`--policy.type=pi0 --robot.type=so100_follower ...`） | argparse + YAML | argparse 但参数名按 `--robot.<field>` 的点号风格写，肌肉记忆通用 |

## 为什么有些东西没有迁移

1. **`eval/eval_pi0.py`** —— lerobot 的 `lerobot-eval` 只跑
   `gym.vector.VectorEnv` 仿真。我们没有仿真环境，而当前的 eval 脚本做的是
   完全不同的事：在 held-out lerobot 数据集上做逐帧开环动作预测。保留。
2. **`train/train_pi0.py`** —— 它已经是 `lerobot.scripts.lerobot_train` 的薄
   包装。直接换成 `lerobot-train --policy.type=pi0 ...` 会丢失 `full|lora|frozen`
   三种模式的预设参数 sweep，包装本身就是价值所在。
3. **老的 `collect/` 工具包** —— 完全保留，向后兼容。新的
   `vla_pi0.scripts.record` 是**并行**路径；两边都写 v3.0 格式，所以历史数据
   集都还能用。
4. **没引入 `draccus`** —— 选择了扁平的 argparse，参数名按 `--robot.<field>`
   的点号风格起，看起来像 draccus 但不依赖。要彻底走 draccus 得给每个脚本写
   一个 config dataclass 加 `@parser.wrap()`，那是更深入的重构；当前形态可以
   后续机械地转过去。

## 注意事项

- `UR7eFollower` 这个 `Robot` 子类**委托**给
  `collect.utils.robot_interface.UR7eInterface` 和
  `collect.utils.robotiq_interface.RobotiqGripper`，并未重写 RTDE。这样
  PolyScope 的 URScript 回放怪癖（30003 端口、`def P1(): ... end` 自动追加
  调用）只在一处维护 —— 见
  [collect/utils/robot_interface.py](../collect/utils/robot_interface.py)。
- `send_action()` 内部就执行**单步关节增量上限**安全裁剪，不在控制循环里做。
  这和 lerobot 把安全检查放在硬件附近的策略一致。上限走 config
  （`max_joint_delta_rad`、`clamp_mode`）。
- 摄像头由 `Robot` 拥有，所以「只开摄像头不连机械臂」的 dry-run 在
  `vla_pi0.scripts.rollout` 暂未支持 —— 想脱机验证模型 + 摄像头时仍然走老的
  `deploy/run_pi0_robot.py --dry-run` 路径最快。

## 接入新机器人

在 `src/vla_pi0/robots/<name>/` 下放一个新包：

```python
# config_<name>.py
@RobotConfig.register_subclass("<name>")
@dataclass(kw_only=True)
class MyRobotConfig(RobotConfig):
    ip: str = "..."

# <name>.py
class MyRobot(Robot):
    config_class = MyRobotConfig
    name = "<name>"
    def connect(self, calibrate=False): ...
    def disconnect(self): ...
    def get_observation(self): ...
    def send_action(self, action): ...
    @property
    def observation_features(self): ...
    @property
    def action_features(self): ...
```

注册之后，每个 `vla_pi0.scripts.*` 入口都能用 `--robot.type=<name> ...`
切过去（前提是把 rollout/record 脚本里硬写的 `UR7eFollower` 换成
`make_robot_from_config` —— 当前还是写死的）。
