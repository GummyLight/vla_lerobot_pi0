"""Real-robot inference loop for the trained pi0 LoRA policy with Pika gripper.

Loads the same LoRA checkpoint that eval_pi0.py uses, then runs a 30Hz control
loop: capture cameras + joint state -> policy -> send joint target to the arm.

Usage:
    python scripts/run_pi0_robotPika.py --robot-ip 169.254.175.10 --gripper-port /dev/ttyUSB1
"""

from __future__ import annotations

import argparse
import contextlib
import cv2
import json
import os
import select
import sys
import threading
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


# Reuse the loader from eval_pi0.py so policy + processor wiring stays identical.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))  # so `collect.utils.*` resolves
from eval_pi0 import load_policy_with_lora, load_processors  # noqa: E402
from collect.utils.pika_interface import PikaGripper  # noqa: E402


# Defaults used when neither config nor CLI overrides a value.
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/run_pi0_robot.yaml"
DEFAULT_CONTROL_HZ = 25.0
DEFAULT_MAX_JOINT_DELTA_RAD = 0.10
DEFAULT_GRIPPER_THRESHOLD = 0.5
DEFAULT_GRIPPER_PORT = ""  # Empty string triggers auto-detection in PikaGripper
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
    ap.add_argument("--gripper-port", default=None, help="Override robot.gripper_port (e.g. /dev/ttyUSB1). Leave empty for auto-detection.")
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
    default_policy_path = REPO_ROOT / "outputs/train/pi0_dataset_autocon_lora/checkpoints/005000/pretrained_model"
    
    if args.policy_path:
        args.policy_path = Path(args.policy_path)
    elif "path" in policy_cfg:
        # If config says a path, use it, but check if it exists.
        p = Path(policy_cfg["path"])
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            args.policy_path = p
        else:
            print(f"⚠ path from config not found: {p}. Falling back to default.")
            args.policy_path = default_policy_path
    else:
        args.policy_path = default_policy_path

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
    args.smoothing_alpha = float(control_cfg.get("smoothing_alpha", 1.0))
    args.gripper_smoothing_alpha = float(control_cfg.get("gripper_smoothing_alpha", 1.0))
    args.smoothing_alpha = max(0.0, min(1.0, args.smoothing_alpha))
    args.gripper_smoothing_alpha = max(0.0, min(1.0, args.gripper_smoothing_alpha))
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

    One RealSense (cam_global) is selected by serial number.
    The other RealSense (cam_wrist) is fetched from the PikaGripper.
    """

    WIDTH = 640
    HEIGHT = 480
    FPS = 30

    def __init__(self, global_serial: str | None, gripper: PikaGripper | None):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise ImportError(
                "pyrealsense2 not installed. `pip install pyrealsense2`."
            ) from e
        self.rs = rs
        self.gripper = gripper

        if global_serial is None:
            ctx = rs.context()
            devs = list(ctx.devices)
            if len(devs) < 1:
                raise RuntimeError("need at least 1 RealSense device for cam_global")
            global_serial = devs[0].get_info(rs.camera_info.serial_number)
            print(f"⚠ no cam_global serial given; auto-picking: {global_serial}")

        self.pipe_global = self._open(global_serial)
        # Drop a few warmup frames.
        for _ in range(5):
            self.pipe_global.wait_for_frames()
        
        if self.gripper:
            # PikaGripper handles its own wrist camera warmup on connect.
            print("[Cameras] Wrist camera will be fetched from PikaGripper.")

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
        return np.asanyarray(f.get_data())

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        img_g = self._read_one(self.pipe_global)
        img_w = None
        if self.gripper:
            # BGR from SDK -> RGB for policy
            img_w_bgr = self.gripper.get_wrist_frame()
            if img_w_bgr is not None:
                import cv2
                img_w = cv2.cvtColor(img_w_bgr, cv2.COLOR_BGR2RGB)
        
        if img_w is None:
            # Fallback if wrist camera fails
            img_w = np.zeros_like(img_g)
            
        return img_g, img_w

    def close(self) -> None:
        try:
            self.pipe_global.stop()
        except:
            pass


# ============================================================================
# Robot control for UR + Pika
# ============================================================================
class URRobotPika:
    """Adapter around your UR and Pika gripper.

    Methods:
      - get_joints() -> np.ndarray (6,) in radians
      - get_gripper() -> float in [0, 1]  (0 = open, 1 = closed)
      - send_joint_target(q: np.ndarray (6,))
      - send_gripper(value: float)
    """

    def __init__(self, ip: str, gripper_port: str = "/dev/ttyUSB1",
                 wrist_serial: str | None = None,
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
        
        try:
            self.gripper = PikaGripper(
                port=gripper_port,
                wrist_realsense_serial=wrist_serial,
                wrist_camera_kind="realsense" if wrist_serial else "none",
            )
            self.gripper.connect()
        except Exception as e:
            self.gripper = None
            print(
                "⚠ PikaGripper init/connect failed; continuing without gripper. "
                f"port={gripper_port!r} wrist_serial={wrist_serial!r} err={e}"
            )
        
        self.control_hz = control_hz
        # velocity in m/s, acceleration in m/s^2
        self.servoj = servoj or {"velocity": 0.01, "acceleration": 0.002,
                                 "lookahead_time": 0.1, "gain": 300}

    def get_joints(self) -> np.ndarray:
        return np.array(self.recv.getActualQ(), dtype=np.float32)

    def get_gripper(self) -> float:
        if self.gripper is None:
            return 0.0
        return float(self.gripper.read_position())

    def get_motor_position(self) -> float:
        """Raw motor position in radians."""
        if self.gripper is None:
            return 0.0
        return float(self.gripper.get_motor_position())

    def send_joint_target(self, q: np.ndarray, dt: float | None = None) -> None:
        sj = self.servoj
        t = dt if (dt is not None and dt > 0) else (1.0 / self.control_hz)
        self.ctrl.servoJ(q.tolist(),
                         sj["velocity"], sj["acceleration"],
                         t, sj["lookahead_time"], sj["gain"])

    def send_gripper(self, value: float) -> None:
        if self.gripper is None:
            return
        self.gripper.write_position(value)

    def set_motor_angle(self, rad: float) -> None:
        """Send raw motor position in radians."""
        if self.gripper is None:
            return
        self.gripper.set_motor_angle(rad)

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
            # Add a small delay for UR to clean up
            time.sleep(0.5)
            self.ctrl.disconnect()
            self.recv.disconnect()
        finally:
            if self.gripper is not None:
                self.gripper.disconnect()


def build_observation(
    joints: np.ndarray, gripper: float,
    img_global: np.ndarray, img_wrist: np.ndarray,
    task: str, device: str,
    expected_image_keys: set[str] | None = None,
    expected_state_dim: int | None = None,
) -> dict:
    """Match the dataset format that the policy was trained on."""
    state = np.concatenate([joints, [gripper]], axis=0).astype(np.float32)
    if expected_state_dim is not None and expected_state_dim > 0 and expected_state_dim != state.shape[0]:
        padded = np.zeros((expected_state_dim,), dtype=np.float32)
        n = min(expected_state_dim, state.shape[0])
        padded[:n] = state[:n]
        state = padded

    def to_chw_float01(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t

    img_g = to_chw_float01(img_global).unsqueeze(0).to(device)
    img_w = to_chw_float01(img_wrist).unsqueeze(0).to(device)

    obs: dict[str, Any] = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [task],
    }
    obs["observation.images.cam_global"] = img_g
    obs["observation.images.cam_wrist"] = img_w
    if expected_image_keys:
        if "observation.images.base_0_rgb" in expected_image_keys:
            obs["observation.images.base_0_rgb"] = img_g
        if "observation.images.left_wrist_0_rgb" in expected_image_keys:
            obs["observation.images.left_wrist_0_rgb"] = img_w
        if "observation.images.right_wrist_0_rgb" in expected_image_keys:
            obs["observation.images.right_wrist_0_rgb"] = img_w
    return obs


def _move_tensors_to_device(x: Any, device: str):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_tensors_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in x)
    return x


@contextlib.contextmanager
def _stdin_cbreak_if_tty():
    if sys.platform == "win32" or not sys.stdin.isatty():
        yield
        return
    try:
        import termios
        import tty
    except Exception:
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class _StopOnQ:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
            except Exception:
                time.sleep(0.05)
                continue
            if not r:
                continue
            try:
                ch = sys.stdin.read(1)
            except Exception:
                continue
            if ch in ("q", "Q"):
                self._stop.set()
                try:
                    while True:
                        r2, _, _ = select.select([sys.stdin], [], [], 0.0)
                        if not r2:
                            break
                        ch2 = sys.stdin.read(1)
                        if ch2 in ("\n", "\r", ""):
                            break
                except Exception:
                    pass
                break


def run_rollout(policy, pre, post, cams, robot, task: str, max_seconds: float, args) -> None:
    """One rollout. Caller is responsible for camera/robot lifecycle."""
    # Reset internal action queue so the chunked policy starts fresh.
    underlying = policy.base_model if hasattr(policy, "base_model") else policy
    if hasattr(underlying, "reset"):
        underlying.reset()

    period = 1.0 / args.control_hz
    t0 = time.perf_counter()
    prev_target = None
    prev_gripper_target = None
    prev_loop_dt = period
    n_steps = 0

    print(f"starting rollout: task={task!r}  max={max_seconds}s  hz={args.control_hz:.0f}  "
          f"dry_run={robot is None}")
    if sys.stdin.isatty():
        print("press 'q' to stop this rollout (will keep the process running)")
    stopper = _StopOnQ()
    try:
        with _stdin_cbreak_if_tty():
            stopper.start()
            while time.perf_counter() - t0 < max_seconds:
                if stopper.should_stop():
                    print("rollout stopped by user input ('q').")
                    break
                tick = time.perf_counter()

                if robot is not None:
                    joints = robot.get_joints()
                    gripper = robot.get_gripper()
                else:
                    joints = np.zeros(6, dtype=np.float32)
                    gripper = 0.0
                img_g, img_w = cams.read()

                raw_gripper_rad = robot.get_motor_position() if robot is not None else 0.0
                obs = build_observation(
                    joints,
                    raw_gripper_rad,
                    img_g,
                    img_w,
                    task,
                    args.device,
                    expected_image_keys=getattr(args, "expected_image_keys", None),
                    expected_state_dim=getattr(args, "expected_state_dim", None),
                )
                obs = pre(obs)
                obs = _move_tensors_to_device(obs, args.device)
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

                if prev_target is not None and args.smoothing_alpha < 1.0:
                    target_joints = args.smoothing_alpha * target_joints + (1.0 - args.smoothing_alpha) * prev_target

                if prev_gripper_target is not None and args.gripper_smoothing_alpha < 1.0:
                    target_gripper = (
                        args.gripper_smoothing_alpha * target_gripper
                        + (1.0 - args.gripper_smoothing_alpha) * prev_gripper_target
                    )

                if robot is not None:
                    robot.send_joint_target(target_joints, dt=prev_loop_dt)
                    if target_gripper > 2.0 or target_gripper < -0.5:
                        robot.send_gripper(1.0 if target_gripper > args.gripper_threshold else 0.0)
                    else:
                        robot.set_motor_angle(target_gripper)
                else:
                    if n_steps % 30 == 0:
                        print(f"step {n_steps:4d}  q={np.round(target_joints, 3)}  gripper={target_gripper:.2f}  "
                              f"dt={prev_loop_dt*1000:.0f}ms")

                prev_target = target_joints
                prev_gripper_target = target_gripper
                n_steps += 1

                elapsed = time.perf_counter() - tick
                if elapsed < period:
                    time.sleep(period - elapsed)
                elif elapsed > 1.5 * period:
                    print(f"⚠ step {n_steps}: control loop slow ({elapsed * 1000:.0f}ms > {period * 1000:.0f}ms)")
                prev_loop_dt = max(time.perf_counter() - tick, period)
    finally:
        stopper.request_stop()
        # Stop servoing but keep RTDE up so the next rollout can resume immediately.
        if robot is not None:
            robot.servo_stop()
        dur = time.perf_counter() - t0
        print(f"rollout done. {n_steps} steps in {dur:.1f}s ({n_steps / max(dur, 1e-3):.1f} Hz avg)")


def warmup_policy(policy, pre, post, cams, robot, task: str, device: str, iters: int = 1) -> None:
    if iters <= 0:
        return
    print(f"warming up policy ({iters} iters)...")
    for _ in range(iters):
        if robot is not None:
            joints = robot.get_joints()
            raw_gripper_rad = robot.get_motor_position()
        else:
            joints = np.zeros(6, dtype=np.float32)
            raw_gripper_rad = 0.0
        img_g, img_w = cams.read()
        obs = build_observation(
            joints,
            raw_gripper_rad,
            img_g,
            img_w,
            task,
            device,
            expected_image_keys=getattr(cams, "expected_image_keys", None),
            expected_state_dim=getattr(cams, "expected_state_dim", None),
        )
        obs = pre(obs)
        obs = _move_tensors_to_device(obs, device)
        with torch.no_grad():
            action = policy.select_action(obs)
        action = post(action)
        _ = action.squeeze(0).detach()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    print("warmup done.")


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

    # 1. Pre-flight hardware check before loading the heavy model
    if not args.dry_run:
        print("checking hardware connectivity...")
        # Check UR Robot (Receive + Control)
        try:
            import rtde_receive
            import rtde_control
            
            # Test Receive
            test_recv = rtde_receive.RTDEReceiveInterface(args.robot_ip)
            q = test_recv.getActualQ()
            mode = test_recv.getRobotMode()
            test_recv.disconnect()
            print(f"✓ UR Receive Interface: reachable (q={np.round(q, 3).tolist()}, mode={mode})")
            
            # Test Control
            test_ctrl = rtde_control.RTDEControlInterface(args.robot_ip)
            is_ok = test_ctrl.isConnected()
            # Check if the robot is in a state that allows control
            is_ready = test_ctrl.isProgramRunning()
            test_ctrl.disconnect()
            
            if is_ok:
                print(f"✓ UR Control Interface: connected (program_running={is_ready})")
            else:
                print(f"✗ UR Control Interface: failed to connect.")
                return 1
                
        except Exception as e:
            print(f"✗ UR Robot connectivity check failed: {e}")
            print("  Hint: Check robot IP, ethernet cable, and ensure UR robot is in 'Remote Control' mode.")
            return 1

        # Check Pika Gripper Port
        if args.gripper_port:
            if not os.path.exists(args.gripper_port):
                print(f"✗ Pika Gripper port {args.gripper_port} does not exist.")
                return 1
            print(f"✓ Pika Gripper port {args.gripper_port} found.")
        
        # Check Cameras
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]
            print(f"✓ Found {len(devices)} RealSense devices: {devices}")
            
            if args.cam_global_serial and args.cam_global_serial not in devices:
                print(f"✗ cam_global {args.cam_global_serial} not found in connected devices.")
                return 1
            if args.cam_wrist_serial and args.cam_wrist_serial not in devices:
                print(f"✗ cam_wrist {args.cam_wrist_serial} not found in connected devices.")
                return 1
        except Exception as e:
            print(f"✗ Camera check failed: {e}")
            return 1
        
        print("✓ All hardware checks passed.\n")

    # 2. Loading policy (heavy operation)
    # Force offline mode for environments without internet
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print(f"loading policy from: {args.policy_path}")
    policy = load_policy_with_lora(args.policy_path, args.device)
    print("✓ policy loaded.")
    
    print("loading processors...")
    pre, post = load_processors(args.policy_path, device=args.device)
    print("✓ processors loaded.")

    def _extract_expected_inputs(policy) -> tuple[set[str] | None, int | None]:
        cfg = getattr(policy, "config", None)
        input_features = getattr(cfg, "input_features", None)
        if not isinstance(input_features, dict):
            return None, None
        expected_images = {k for k in input_features.keys() if isinstance(k, str) and k.startswith("observation.images.")}
        state_feat = input_features.get("observation.state")

        def _shape0(feat) -> int | None:
            if feat is None:
                return None
            shape = getattr(feat, "shape", None)
            if shape:
                return int(shape[0])
            if isinstance(feat, dict):
                shp = feat.get("shape")
                if shp:
                    return int(shp[0])
            return None

        return (expected_images or None), _shape0(state_feat)

    args.expected_image_keys, args.expected_state_dim = _extract_expected_inputs(policy)

    print("initializing cameras...")
    cams = Cameras(args.cam_global_serial, None)
    cams.expected_image_keys = args.expected_image_keys
    cams.expected_state_dim = args.expected_state_dim

    try:
        print("initializing robot for first rollout...")
        robot = None if args.dry_run else URRobotPika(
            args.robot_ip,
            gripper_port=args.gripper_port,
            wrist_serial=args.cam_wrist_serial,
            control_hz=args.control_hz,
            servoj=args.servoj,
        )
        if robot:
            cams.gripper = robot.gripper

        warmup_policy(policy, pre, post, cams, robot, args.task, args.device, iters=1)

        run_rollout(policy, pre, post, cams, robot, args.task, args.max_seconds, args)

        if args.loop:
            task, secs = args.task, args.max_seconds
            while True:
                nxt = _prompt_next_rollout(task, secs)
                if nxt is None:
                    print("exiting loop mode.")
                    break
                task, secs = nxt

                print("\nreconnecting robot for next rollout (allows manual adjustment)...")
                if robot:
                    robot.close()
                
                robot = None if args.dry_run else URRobotPika(
                    args.robot_ip,
                    gripper_port=args.gripper_port,
                    wrist_serial=args.cam_wrist_serial,
                    control_hz=args.control_hz,
                    servoj=args.servoj,
                )
                if robot:
                    cams.gripper = robot.gripper

                warmup_policy(policy, pre, post, cams, robot, task, args.device, iters=1)

                run_rollout(policy, pre, post, cams, robot, task, secs, args)
    finally:
        if robot is not None:
            robot.close()
        cams.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
