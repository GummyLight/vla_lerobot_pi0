# Aligning this project with `huggingface/lerobot`

> 中文版见 [lerobot_alignment_cn.md](lerobot_alignment_cn.md).

This project sits *next to* the upstream [`lerobot`](https://github.com/huggingface/lerobot)
codebase: training and dataset loading already use lerobot's CLI/dataset
classes, but data collection, real-robot inference and config plumbing
were written from scratch around the legacy `collect/` toolkit. This
document records the gap between the two stacks and the alignment work
that's been done on the `align-with-lerobot` branch.

## TL;DR — what changed

A new `src/vla_pi0/` package mirrors lerobot's layout:

```
src/vla_pi0/
├── robots/
│   └── ur7e_follower/                 # UR7e + Robotiq 2F-58 follower
│       ├── config_ur7e_follower.py    # @RobotConfig.register_subclass("ur7e_follower")
│       └── ur7e_follower.py           # UR7eFollower(Robot)
└── scripts/
    ├── record.py                      # mirror of lerobot-record
    └── rollout.py                     # mirror of lerobot-rollout
```

After alignment you can drive the rig the same way the upstream SO-100 / Koch
demos do:

```bash
# Closed-loop policy inference (was: scripts/run_pi0_robot.py)
python -m vla_pi0.scripts.rollout \
    --policy-path outputs/train/pi0_3d_printer_lora/checkpoints/last/pretrained_model \
    --task "open the 3D printer" \
    --robot.ip 192.168.1.100 \
    --robot.cameras.cam-global.serial <SN_A> \
    --robot.cameras.cam-wrist.serial  <SN_B>

# Demonstration recording (was: collect/collect_urscript.py / collect_pika.py)
python -m vla_pi0.scripts.record \
    --dataset-name my_demo \
    --task "open the 3D printer" \
    --source urscript \
    --urscript "collect/urscripts/open the 3D printer.script" \
    --robot.ip 192.168.1.100
```

The legacy scripts under `collect/` and `scripts/run_pi0_robot.py` still
work unchanged — the new package is additive.

## Side-by-side comparison

| Concern | Upstream `lerobot` | This project (before) | This project (now) |
|---|---|---|---|
| Robot abstraction | `lerobot.robots.robot.Robot` ABC with `connect/disconnect/get_observation/send_action/observation_features/action_features/is_connected/is_calibrated` | none — raw `UR7eInterface`, `RobotiqGripper`, `MultiCamera` mixed into each script | `vla_pi0.robots.ur7e_follower.UR7eFollower(Robot)` with full contract |
| Robot config | `RobotConfig` (`draccus.ChoiceRegistry`) + `@RobotConfig.register_subclass("name")` | YAML files (`configs/run_pi0_robot.yaml`) parsed ad-hoc | `UR7eFollowerConfig` registered as `ur7e_follower`; legacy YAML still consumed by `to_legacy_dict()` |
| Camera abstraction | `lerobot.cameras.realsense.RealSenseCamera` (registered as `intelrealsense`) | inline `pyrealsense2` calls in two places (`collect/utils/camera_interface.py` + `scripts/run_pi0_robot.py`) | one place — `UR7eFollower._open_cameras()` |
| Data recording | `lerobot-record` → `make_robot_from_config(...)` + `LeRobotDataset` writer | `collect/collect_urscript.py` + `collect/collect_pika.py` (own writer) | `vla_pi0.scripts.record` reuses `LeRobotWriter` *via* the `Robot` contract |
| Real-robot inference | `lerobot-rollout` (with `strategy.type=base|sentry|...`) | `scripts/run_pi0_robot.py` (custom Cameras + URRobot classes) | `vla_pi0.scripts.rollout` (Robot abstraction; strategy-shaped loop) |
| Training | `lerobot-train` (`TrainPipelineConfig`) | `scripts/train_pi0.py` wrapping `lerobot.scripts.lerobot_train` | unchanged — already aligned |
| Offline eval | `lerobot-eval` (sim only, gym vector env) | `scripts/eval_pi0.py` (offline open-loop on dataset) | unchanged — covers the case lerobot-eval doesn't (real-data offline) |
| Dataset format | LeRobot v3.0 (parquet + mp4, multi-episode chunks) | LeRobot v3.0 | unchanged — already aligned |
| CLI configs | `draccus` (`--policy.type=pi0 --robot.type=so100_follower ...`) | argparse + YAML | argparse with `--robot.<field>` dotted names that match draccus muscle memory |

## What's intentionally NOT migrated

1. **`scripts/eval_pi0.py`** — lerobot's `lerobot-eval` only runs against
   `gym.vector.VectorEnv` simulators. We don't have a sim, and the existing
   eval script does something genuinely different: open-loop frame-by-frame
   action prediction against a held-out lerobot dataset. Keep it.
2. **`scripts/train_pi0.py`** — already a thin wrapper around
   `lerobot.scripts.lerobot_train`. Replacing it with raw
   `lerobot-train --policy.type=pi0 ...` invocations would lose the
   per-method (`full|lora|frozen`) defaults sweep; the wrapper is the
   value.
3. **Legacy `collect/` toolkit** — kept as-is for backwards compat. The new
   `vla_pi0.scripts.record` is a *parallel* path; both write the same v3.0
   format so existing datasets are still loadable.
4. **`draccus` CLI** — opted for a flat argparse with `--robot.<field>`
   names that resemble draccus's nested syntax. Going full draccus needs
   a deeper refactor (config dataclass per script + `@parser.wrap()`); the
   shape we have here is mechanically convertable later.

## Gotchas

- The `UR7eFollower` `Robot` subclass *delegates* to
  `collect.utils.robot_interface.UR7eInterface` and
  `collect.utils.robotiq_interface.RobotiqGripper` rather than reimplementing
  RTDE. That keeps the Windows-specific URScript replay quirks (port 30003,
  PolyScope `def P1(): ... end` auto-call) in one place — see
  [collect/utils/robot_interface.py](../collect/utils/robot_interface.py).
- `send_action()` runs the per-step joint-delta safety cap *inside* the
  robot, not in the loop. That mirrors lerobot's policy of putting safety
  next to hardware. The cap is a config knob (`max_joint_delta_rad`,
  `clamp_mode`).
- Cameras are owned by the `Robot`, so a dry-run with cameras-only is not
  yet supported in `vla_pi0.scripts.rollout` — the legacy
  `scripts/run_pi0_robot.py --dry-run` path is still the easy way to
  smoke-test the model + cameras without RTDE.

## Adding a different robot

Drop a new package under `src/vla_pi0/robots/<name>/`:

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

Once registered, every `vla_pi0.scripts.*` entrypoint can target it via
`--robot.type=<name> ...` (after we wire the rollout/record scripts to
`make_robot_from_config` — currently they hard-code `UR7eFollower`).
