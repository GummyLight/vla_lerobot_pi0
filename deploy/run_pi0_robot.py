"""Real-robot inference loop for the trained pi0 LoRA policy.

Loads the same LoRA checkpoint that eval_pi0.py uses, then runs a 30Hz control
loop: capture cameras + joint state -> policy -> send joint target to the arm.

⚠ READ BEFORE RUNNING ON A REAL ROBOT ⚠
- First run: limit the UR teach pendant speed slider to 20-30%.
- Have an e-stop within reach.
- Clear the workspace except the 3D printer.
- The script enforces a per-step joint delta safety cap; tune it for your arm.
- Cameras MUST be in the same poses + same resolutions as the training dataset
  (cam_global + cam_wrist). Pose drift = severe degradation.

Usage:
    # All hardware params live in deploy/configs/run_pi0_robot.yaml (default config path).
    python deploy/run_pi0_robot.py                            # closed-loop on real robot
    python deploy/run_pi0_robot.py --dry-run                  # no actions sent
    python deploy/run_pi0_robot.py --task "close the 3D printer" --max-seconds 15
    python deploy/run_pi0_robot.py --config deploy/configs/other.yaml

Any CLI flag overrides the matching value in the config file.

Hardware assumptions (filled in for the current rig):
- Two Intel RealSense D435i/D455 (one as cam_global, one as cam_wrist), 640x480 RGB @ 30Hz.
  Find serials with `rs-enumerate-devices` or pyrealsense2 `rs.context().devices`.
- Robotiq 2F-85/140 plugged into the UR control box; the UR-side URCap exposes
  the gripper on tcp://<robot_ip>:63352. We talk to it directly over that socket.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if sys.platform == "win32":
    os.environ.setdefault("HF_HOME", r"D:\.hfcache")
else:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))


# Reuse the loader from eval/eval_pi0.py so policy + processor wiring stays identical.
sys.path.insert(0, str(REPO_ROOT / "eval"))
sys.path.insert(0, str(REPO_ROOT))  # so `collect.utils.*` resolves
from eval_pi0 import load_policy_with_lora, load_processors  # noqa: E402
from collect.utils.robotiq_interface import RobotiqGripper  # noqa: E402


# Defaults used when neither config nor CLI overrides a value.
DEFAULT_CONFIG_PATH = REPO_ROOT / "deploy/configs/run_pi0_robot.yaml"
DEFAULT_CONTROL_HZ = 30.0
DEFAULT_MAX_JOINT_DELTA_RAD = 0.10
DEFAULT_GRIPPER_THRESHOLD = 0.5
DEFAULT_GRIPPER_PORT = 63352
DEFAULT_MAX_SECONDS = 30.0


def _load_config_file(path: Path) -> dict[str, Any]:
    """Load yaml or json based on file extension."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML not installed. `pip install pyyaml`.") from e
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _find_camera(cams_list: list[dict], name: str) -> dict | None:
    for c in cams_list or []:
        if c.get("name") == name:
            return c
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                    help=f"YAML/JSON config (default: {DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)}). "
                         "CLI args override config values.")
    ap.add_argument("--policy-path", type=Path, default=None,
                    help="Override policy.path from config.")
    ap.add_argument("--task", default=None, help="Override task string from config.")
    ap.add_argument("--robot-ip", default=None, help="Override robot.ip from config.")
    ap.add_argument("--cam-global-serial", default=None, help="Override cameras[cam_global].serial.")
    ap.add_argument("--cam-wrist-serial", default=None, help="Override cameras[cam_wrist].serial.")
    ap.add_argument("--gripper-port", type=int, default=None, help="Override robot.gripper_port.")
    ap.add_argument("--max-seconds", type=float, default=None, help="Override control.max_seconds.")
    ap.add_argument("--device", default=None, help="Override policy.device (cuda/cpu).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the loop but DO NOT send actions to the robot. Prints them instead.")
    ap.add_argument("--loop", action="store_true",
                    help="After each rollout, prompt for the next task/duration. "
                         "Model + cameras + robot stay loaded across rollouts.")

    args = ap.parse_args()

    # Load config (default path is fine if it exists; otherwise CLI must supply everything).
    cfg: dict[str, Any] = {}
    if args.config and args.config.exists():
        cfg = _load_config_file(args.config)
        print(f"loaded config: {args.config}")
    elif args.config != DEFAULT_CONFIG_PATH:
        # User explicitly pointed at a missing file — that's a real error.
        raise FileNotFoundError(f"config file not found: {args.config}")

    policy_cfg = cfg.get("policy") or {}
    robot_cfg = cfg.get("robot") or {}
    servoj_cfg = robot_cfg.get("servoj") or {}
    control_cfg = cfg.get("control") or {}
    cams_list = cfg.get("cameras") or []
    cam_global = _find_camera(cams_list, "cam_global") or {}
    cam_wrist = _find_camera(cams_list, "cam_wrist") or {}

    # CLI > config > built-in default. Attach everything onto args for downstream use.
    args.policy_path = args.policy_path or (Path(policy_cfg["path"]) if "path" in policy_cfg else
        REPO_ROOT / "outputs/train/pi0_3d_printer_lora/checkpoints/005000/pretrained_model")
    args.task = args.task or cfg.get("task")
    args.robot_ip = args.robot_ip or robot_cfg.get("ip")
    args.gripper_port = args.gripper_port or robot_cfg.get("gripper_port") or DEFAULT_GRIPPER_PORT
    args.cam_global_serial = args.cam_global_serial or cam_global.get("serial")
    args.cam_wrist_serial = args.cam_wrist_serial or cam_wrist.get("serial")
    args.max_seconds = args.max_seconds or control_cfg.get("max_seconds") or DEFAULT_MAX_SECONDS
    args.device = args.device or policy_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    args.control_hz = float(control_cfg.get("hz") or DEFAULT_CONTROL_HZ)
    args.max_joint_delta_rad = float(control_cfg.get("max_joint_delta_rad") or DEFAULT_MAX_JOINT_DELTA_RAD)
    args.gripper_threshold = float(control_cfg.get("gripper_threshold") or DEFAULT_GRIPPER_THRESHOLD)
    args.clamp_mode = (control_cfg.get("clamp_mode") or "refuse").lower()
    if args.clamp_mode not in ("refuse", "clip"):
        raise ValueError(f"clamp_mode must be 'refuse' or 'clip', got {args.clamp_mode!r}")
    args.servoj = {
        "velocity": float(servoj_cfg.get("velocity", 0.5)),
        "acceleration": float(servoj_cfg.get("acceleration", 0.5)),
        "lookahead_time": float(servoj_cfg.get("lookahead_time", 0.1)),
        "gain": int(servoj_cfg.get("gain", 300)),
    }

    if not args.dry_run and not args.robot_ip:
        raise ValueError("robot.ip is required for live runs (set in config or pass --robot-ip).")
    if not args.task:
        raise ValueError("task is required (set in config or pass --task).")

    return args


# ============================================================================
# RealSense camera capture (cam_global + cam_wrist)
# ============================================================================
class Cameras:
    """Two Intel RealSense streams returning uint8 H×W×3 RGB arrays.

    Each RealSense is selected by serial number so the global/wrist mapping
    survives reboots and re-plugging. Pass --cam-*-serial on the CLI; if
    omitted, we fall back to the first/second enumerated device (fragile —
    only fine for first-time poking).
    """

    WIDTH = 640
    HEIGHT = 480
    FPS = 30

    def __init__(self, global_serial: str | None, wrist_serial: str | None):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise ImportError(
                "pyrealsense2 not installed. `pip install pyrealsense2`."
            ) from e
        from concurrent.futures import ThreadPoolExecutor
        self.rs = rs
        # Two threads so the per-camera `wait_for_frames` calls run in parallel
        # (otherwise serial blocking dominates the control-loop period).
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rs")

        if global_serial is None or wrist_serial is None:
            ctx = rs.context()
            devs = list(ctx.devices)
            if len(devs) < 2:
                raise RuntimeError(
                    f"need 2 RealSense devices, found {len(devs)}. "
                    "Plug both in, or pass explicit --cam-*-serial."
                )
            serials = [d.get_info(rs.camera_info.serial_number) for d in devs]
            print(f"⚠ no serials given; auto-picking from enumeration order: {serials}")
            if global_serial is None:
                global_serial = serials[0]
            if wrist_serial is None:
                wrist_serial = serials[1] if serials[1] != global_serial else serials[0]

        if global_serial == wrist_serial:
            raise ValueError("cam_global and cam_wrist serials must differ")

        self.pipe_global = self._open(global_serial)
        self.pipe_wrist = self._open(wrist_serial)
        # Drop a few warmup frames so AE/AWB stabilise before we start the loop.
        for _ in range(5):
            self.pipe_global.wait_for_frames()
            self.pipe_wrist.wait_for_frames()

    def _open(self, serial: str):
        rs = self.rs
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, self.WIDTH, self.HEIGHT, rs.format.rgb8, self.FPS)
        pipe = rs.pipeline()
        pipe.start(cfg)
        return pipe

    @staticmethod
    def _read_one(pipe) -> np.ndarray:
        f = pipe.wait_for_frames().get_color_frame()
        if not f:
            raise RuntimeError("RealSense returned empty color frame")
        # rgb8 -> already RGB, matches dataset.
        return np.asanyarray(f.get_data())

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        fut_g = self._executor.submit(self._read_one, self.pipe_global)
        fut_w = self._executor.submit(self._read_one, self.pipe_wrist)
        # max(t_g, t_w) instead of t_g + t_w.
        return fut_g.result(), fut_w.result()

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        finally:
            try:
                self.pipe_global.stop()
            finally:
                self.pipe_wrist.stop()


# ============================================================================
# === HARDWARE_TODO: Robot control
# ============================================================================
class URRobot:
    """Adapter around your UR. Default uses `ur_rtde`.

    Methods:
      - get_joints() -> np.ndarray (6,) in radians
      - get_gripper() -> float in [0, 1]  (0 = open, 1 = closed)
      - send_joint_target(q: np.ndarray (6,))
      - send_gripper(value: float)
    """

    def __init__(self, ip: str, gripper_port: int = 63352,
                 control_hz: float = 30.0, servoj: dict | None = None):
        try:
            import rtde_control
            import rtde_receive
        except ImportError as e:
            raise ImportError(
                "ur_rtde not installed. `pip install ur_rtde` and ensure UR controller is reachable."
            ) from e

        self.ctrl = rtde_control.RTDEControlInterface(ip)
        self.recv = rtde_receive.RTDEReceiveInterface(ip)
        # Shared with collect/utils/robotiq_interface.py — needs an explicit
        # connect() (constructor only stores host/port).
        self.gripper = RobotiqGripper(ip, port=gripper_port)
        self.gripper.connect()
        self.control_hz = control_hz
        self.servoj = servoj or {"velocity": 0.5, "acceleration": 0.5,
                                 "lookahead_time": 0.1, "gain": 300}

    def get_joints(self) -> np.ndarray:
        return np.array(self.recv.getActualQ(), dtype=np.float32)

    def get_gripper(self) -> float:
        if self.gripper is None:
            return 0.0
        return float(self.gripper.read_position())

    def send_joint_target(self, q: np.ndarray, dt: float | None = None) -> None:
        # servoJ: realtime joint servo control. `time` should match the actual
        # loop period — if we send commands at 78ms intervals but tell servoJ
        # to reach the target in 33ms, the arm jerks (sprint then idle).
        # ur_rtde's servoJ takes positional args only:
        #   (q, velocity, acceleration, time, lookahead_time, gain)
        sj = self.servoj
        t = dt if (dt is not None and dt > 0) else (1.0 / self.control_hz)
        self.ctrl.servoJ(q.tolist(),
                         sj["velocity"], sj["acceleration"],
                         t, sj["lookahead_time"], sj["gain"])

    def send_gripper(self, value: float) -> None:
        if self.gripper is None:
            return
        self.gripper.write_position(value)

    def servo_stop(self) -> None:
        """Stop servo motion but keep the RTDE connection alive."""
        try:
            self.ctrl.servoStop()
        except Exception as e:
            print(f"⚠ servoStop failed: {e}")

    def close(self) -> None:
        try:
            self.ctrl.servoStop()
            self.ctrl.stopScript()
        finally:
            if self.gripper is not None:
                # On collect's RobotiqGripper, .close() actuates the jaws —
                # use .disconnect() to drop the socket without moving anything.
                self.gripper.disconnect()


def build_observation(
    joints: np.ndarray, gripper: float,
    img_global: np.ndarray, img_wrist: np.ndarray,
    task: str, device: str,
) -> dict:
    """Match the dataset format that the policy was trained on."""
    # state = 6 joints + gripper, float32
    state = np.concatenate([joints, [gripper]], axis=0).astype(np.float32)

    def to_chw_float01(img: np.ndarray) -> torch.Tensor:
        # HxWxC uint8 [0,255] -> CxHxW float32 [0,1]
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t

    obs = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.images.cam_global": to_chw_float01(img_global).unsqueeze(0).to(device),
        "observation.images.cam_wrist":  to_chw_float01(img_wrist).unsqueeze(0).to(device),
        "task": [task],
    }
    return obs


def run_rollout(policy, pre, post, cams, robot, task: str, max_seconds: float, args) -> None:
    """One rollout. Caller is responsible for camera/robot lifecycle."""
    # Reset internal action queue so the chunked policy starts fresh.
    underlying = policy.base_model if hasattr(policy, "base_model") else policy
    if hasattr(underlying, "reset"):
        underlying.reset()

    period = 1.0 / args.control_hz
    t0 = time.perf_counter()
    prev_target = None
    prev_loop_dt = period
    n_steps = 0

    print(f"starting rollout: task={task!r}  max={max_seconds}s  hz={args.control_hz:.0f}  "
          f"dry_run={robot is None}")
    try:
        while time.perf_counter() - t0 < max_seconds:
            tick = time.perf_counter()

            if robot is not None:
                joints = robot.get_joints()
                gripper = robot.get_gripper()
            else:
                joints = np.zeros(6, dtype=np.float32)
                gripper = 0.0
            img_g, img_w = cams.read()

            obs = build_observation(joints, gripper, img_g, img_w, task, args.device)
            obs = pre(obs)
            with torch.no_grad():
                action = policy.select_action(obs)
            action = post(action)
            action = action.squeeze(0).detach().cpu().float().numpy()

            target_joints = action[:6]
            target_gripper = float(action[6])

            if prev_target is not None:
                raw_delta = target_joints - prev_target
                max_abs = np.abs(raw_delta).max()
                if max_abs > args.max_joint_delta_rad:
                    if args.clamp_mode == "clip":
                        # Step max_joint_delta_rad in the predicted direction.
                        scale = args.max_joint_delta_rad / max_abs
                        target_joints = prev_target + raw_delta * scale
                        if n_steps % 30 == 0:
                            print(f"  step {n_steps}: clipped raw delta {max_abs:.3f} → "
                                  f"{args.max_joint_delta_rad:.3f} rad")
                    else:
                        print(f"⚠ step {n_steps}: max joint delta {max_abs:.3f} rad > "
                              f"{args.max_joint_delta_rad}; refusing to send. Holding previous target.")
                        target_joints = prev_target

            if robot is not None:
                robot.send_joint_target(target_joints, dt=prev_loop_dt)
                robot.send_gripper(1.0 if target_gripper > args.gripper_threshold else 0.0)
            else:
                if n_steps % 30 == 0:
                    print(f"step {n_steps:4d}  q={np.round(target_joints, 3)}  gripper={target_gripper:.2f}  "
                          f"dt={prev_loop_dt*1000:.0f}ms")

            prev_target = target_joints
            n_steps += 1

            elapsed = time.perf_counter() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
            elif elapsed > 1.5 * period:
                print(f"⚠ step {n_steps}: control loop slow ({elapsed * 1000:.0f}ms > {period * 1000:.0f}ms)")
            prev_loop_dt = max(time.perf_counter() - tick, period)
    finally:
        # Stop servoing but keep RTDE up so the next rollout can resume immediately.
        if robot is not None:
            robot.servo_stop()
        dur = time.perf_counter() - t0
        print(f"rollout done. {n_steps} steps in {dur:.1f}s ({n_steps / max(dur, 1e-3):.1f} Hz avg)")


def _prompt_next_rollout(default_task: str, default_seconds: float) -> tuple[str, float] | None:
    """Returns (task, max_seconds) or None to quit. Empty input keeps defaults."""
    print()
    print(f"--- next rollout (defaults: task={default_task!r}, max_seconds={default_seconds}) ---")
    print("  enter to repeat with defaults | type a new task | 'q' to quit")
    line = input("task> ").strip()
    if line.lower() in ("q", "quit", "exit"):
        return None
    task = line if line else default_task
    sec_line = input(f"max_seconds [{default_seconds}]> ").strip()
    try:
        seconds = float(sec_line) if sec_line else default_seconds
    except ValueError:
        print(f"⚠ couldn't parse {sec_line!r} as float; using {default_seconds}")
        seconds = default_seconds
    return task, seconds


def main() -> int:
    args = parse_args()

    print(f"loading policy from: {args.policy_path}")
    policy = load_policy_with_lora(args.policy_path, args.device)
    pre, post = load_processors(args.policy_path)

    cams = Cameras(args.cam_global_serial, args.cam_wrist_serial)
    robot = None if args.dry_run else URRobot(
        args.robot_ip,
        gripper_port=args.gripper_port,
        control_hz=args.control_hz,
        servoj=args.servoj,
    )

    try:
        run_rollout(policy, pre, post, cams, robot, args.task, args.max_seconds, args)

        if args.loop:
            task, secs = args.task, args.max_seconds
            while True:
                nxt = _prompt_next_rollout(task, secs)
                if nxt is None:
                    print("exiting loop mode.")
                    break
                task, secs = nxt
                run_rollout(policy, pre, post, cams, robot, task, secs, args)
    finally:
        if robot is not None:
            robot.close()
        cams.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
