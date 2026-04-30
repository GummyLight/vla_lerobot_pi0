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
    python scripts/run_pi0_robot.py \\
        --policy-path outputs/train/pi0_3d_printer_lora/checkpoints/005000/pretrained_model \\
        --task "open the 3D printer" \\
        --robot-ip 192.168.1.100 \\
        --cam-global-serial 1234567890 \\
        --cam-wrist-serial 0987654321 \\
        --max-seconds 30

Hardware assumptions (filled in for the current rig):
- Two Intel RealSense D435i/D455 (one as cam_global, one as cam_wrist), 640x480 RGB @ 30Hz.
  Find serials with `rs-enumerate-devices` or pyrealsense2 `rs.context().devices`.
- Robotiq 2F-85/140 plugged into the UR control box; the UR-side URCap exposes
  the gripper on tcp://<robot_ip>:63352. We talk to it directly over that socket.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if sys.platform == "win32":
    os.environ.setdefault("HF_HOME", r"D:\.hfcache")
else:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))


# Reuse the loader from eval_pi0.py so policy + processor wiring stays identical.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from eval_pi0 import load_policy_with_lora, load_processors  # noqa: E402


CONTROL_HZ = 30.0          # must match dataset fps
MAX_JOINT_DELTA_RAD = 0.10 # per-step cap; if predicted target jumps more than this, refuse
GRIPPER_THRESHOLD = 0.5    # >this means closed; <this means open. Tune to your gripper.


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", type=Path,
                    default=REPO_ROOT / "outputs/train/pi0_3d_printer_lora/checkpoints/005000/pretrained_model")
    ap.add_argument("--task", required=True, help="Language instruction, e.g. 'open the 3D printer'")
    ap.add_argument("--robot-ip", required=True, help="UR controller IP")
    ap.add_argument("--cam-global-serial", default=None,
                    help="RealSense serial for cam_global. If omitted, picks the first device.")
    ap.add_argument("--cam-wrist-serial", default=None,
                    help="RealSense serial for cam_wrist. If omitted, picks the second device.")
    ap.add_argument("--gripper-port", type=int, default=63352,
                    help="UR-side Robotiq URCap socket port (default 63352).")
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="Hard cutoff for the rollout (safety).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the loop but DO NOT send actions to the robot. Prints them instead.")
    return ap.parse_args()


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
        self.rs = rs

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

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        # wait_for_frames blocks up to 5s by default; that's fine at 30Hz.
        f_g = self.pipe_global.wait_for_frames().get_color_frame()
        f_w = self.pipe_wrist.wait_for_frames().get_color_frame()
        if not f_g or not f_w:
            raise RuntimeError("RealSense returned empty color frame")
        # rgb8 -> already RGB, matches dataset.
        return np.asanyarray(f_g.get_data()), np.asanyarray(f_w.get_data())

    def close(self) -> None:
        try:
            self.pipe_global.stop()
        finally:
            self.pipe_wrist.stop()


# ============================================================================
# Robotiq 2F-85/140 over UR's URCap socket (port 63352)
# ============================================================================
class RobotiqGripper:
    """Minimal client for the Robotiq URCap socket on the UR controller.

    The URCap exposes a line-based protocol on tcp://<robot_ip>:63352:
        SET POS <0..255>\n  -> 'ack\n'
        GET POS\n           -> 'POS <0..255>\n'
        SET ACT 1\n         -> activates the gripper if not yet activated
        SET GTO 1\n         -> enables go-to-position
    Position is the gripper's internal scale: 0 = fully open, 255 = fully closed.
    """

    def __init__(self, ip: str, port: int = 63352, timeout: float = 2.0):
        import socket
        self._socket = socket
        self.sock = socket.create_connection((ip, port), timeout=timeout)
        self.sock.settimeout(timeout)
        # Activate + enable GTO. Idempotent: re-running on an already-active
        # gripper is harmless. If your URCap auto-activates on URe-boot you
        # can drop these, but they are cheap.
        self._cmd("SET ACT 1")
        self._cmd("SET GTO 1")
        # Reasonable default speed/force; tune to your task.
        self._cmd("SET SPE 200")
        self._cmd("SET FOR 100")

    def _cmd(self, line: str) -> str:
        self.sock.sendall((line + "\n").encode("ascii"))
        return self.sock.recv(1024).decode("ascii", errors="replace").strip()

    def read_position(self) -> float:
        """Return current position in [0, 1]: 0=open, 1=closed (matches dataset)."""
        reply = self._cmd("GET POS")
        # Reply format: 'POS <int>'
        try:
            raw = int(reply.split()[-1])
        except (ValueError, IndexError):
            return 0.0
        return max(0.0, min(1.0, raw / 255.0))

    def write_position(self, value: float) -> None:
        """Send a position in [0, 1]: 0=open, 1=closed."""
        raw = int(round(max(0.0, min(1.0, value)) * 255))
        self._cmd(f"SET POS {raw}")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


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

    def __init__(self, ip: str, gripper_port: int = 63352):
        try:
            import rtde_control
            import rtde_receive
        except ImportError as e:
            raise ImportError(
                "ur_rtde not installed. `pip install ur_rtde` and ensure UR controller is reachable."
            ) from e

        self.ctrl = rtde_control.RTDEControlInterface(ip)
        self.recv = rtde_receive.RTDEReceiveInterface(ip)
        self.gripper = RobotiqGripper(ip, port=gripper_port)

    def get_joints(self) -> np.ndarray:
        return np.array(self.recv.getActualQ(), dtype=np.float32)

    def get_gripper(self) -> float:
        if self.gripper is None:
            return 0.0
        return float(self.gripper.read_position())

    def send_joint_target(self, q: np.ndarray) -> None:
        # servoJ: realtime joint servo control. Args tuned for ~30Hz control.
        # (lookahead_time=0.1, gain=300 are reasonable starting points.)
        self.ctrl.servoJ(q.tolist(), velocity=0.5, acceleration=0.5,
                         time=1.0 / CONTROL_HZ, lookahead_time=0.1, gain=300)

    def send_gripper(self, value: float) -> None:
        if self.gripper is None:
            return
        self.gripper.write_position(value)

    def close(self) -> None:
        try:
            self.ctrl.servoStop()
            self.ctrl.stopScript()
        finally:
            if self.gripper is not None:
                self.gripper.close()


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


def main() -> int:
    args = parse_args()

    print(f"loading policy from: {args.policy_path}")
    policy = load_policy_with_lora(args.policy_path, args.device)
    pre, post = load_processors(args.policy_path)

    cams = Cameras(args.cam_global_serial, args.cam_wrist_serial)
    robot = None if args.dry_run else URRobot(args.robot_ip, gripper_port=args.gripper_port)

    # Reset internal action queue so the chunked policy starts fresh.
    underlying = policy.base_model if hasattr(policy, "base_model") else policy
    if hasattr(underlying, "reset"):
        underlying.reset()

    period = 1.0 / CONTROL_HZ
    t0 = time.perf_counter()
    prev_target = None
    n_steps = 0

    print(f"starting rollout, max {args.max_seconds}s, {CONTROL_HZ:.0f}Hz, dry_run={args.dry_run}")
    try:
        while time.perf_counter() - t0 < args.max_seconds:
            tick = time.perf_counter()

            # 1. observe
            if robot is not None:
                joints = robot.get_joints()
                gripper = robot.get_gripper()
            else:
                joints = np.zeros(6, dtype=np.float32)  # dry-run dummy
                gripper = 0.0
            img_g, img_w = cams.read()

            # 2. inference
            obs = build_observation(joints, gripper, img_g, img_w, args.task, args.device)
            obs = pre(obs)
            with torch.no_grad():
                action = policy.select_action(obs)
            action = post(action)
            action = action.squeeze(0).detach().cpu().float().numpy()  # (7,)

            target_joints = action[:6]
            target_gripper = float(action[6])

            # 3. safety: clamp jump
            if prev_target is not None:
                delta = np.abs(target_joints - prev_target).max()
                if delta > MAX_JOINT_DELTA_RAD:
                    print(f"⚠ step {n_steps}: max joint delta {delta:.3f} rad > {MAX_JOINT_DELTA_RAD}; "
                          f"refusing to send. Holding previous target.")
                    target_joints = prev_target

            # 4. send
            if robot is not None:
                robot.send_joint_target(target_joints)
                robot.send_gripper(1.0 if target_gripper > GRIPPER_THRESHOLD else 0.0)
            else:
                if n_steps % 30 == 0:
                    print(f"step {n_steps:4d}  q={np.round(target_joints, 3)}  gripper={target_gripper:.2f}")

            prev_target = target_joints
            n_steps += 1

            # 5. pace
            elapsed = time.perf_counter() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
            else:
                if elapsed > 1.5 * period:
                    print(f"⚠ step {n_steps}: control loop slow ({elapsed * 1000:.0f}ms > {period * 1000:.0f}ms)")
    finally:
        if robot is not None:
            robot.close()
        cams.close()
        dur = time.perf_counter() - t0
        print(f"rollout done. {n_steps} steps in {dur:.1f}s ({n_steps / max(dur, 1e-3):.1f} Hz avg)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
