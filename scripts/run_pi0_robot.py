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
        --max-seconds 30
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
    ap.add_argument("--max-seconds", type=float, default=30.0,
                    help="Hard cutoff for the rollout (safety).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the loop but DO NOT send actions to the robot. Prints them instead.")
    return ap.parse_args()


# ============================================================================
# === HARDWARE_TODO: Camera capture
# ============================================================================
class Cameras:
    """Adapter around your two cameras. Returns frames as uint8 HxWxC RGB arrays.

    Default impl uses OpenCV `cv2.VideoCapture`. Replace with pyrealsense2 etc.
    if you use Intel RealSense / Zed / GigE cameras.
    """

    def __init__(self, global_idx: int = 0, wrist_idx: int = 1, height: int = 480, width: int = 640):
        import cv2
        self.cv2 = cv2
        self.cap_global = cv2.VideoCapture(global_idx)
        self.cap_wrist = cv2.VideoCapture(wrist_idx)
        for cap in (self.cap_global, self.cap_wrist):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap_global.isOpened() or not self.cap_wrist.isOpened():
            raise RuntimeError("could not open both cameras")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        ok1, frame_global = self.cap_global.read()
        ok2, frame_wrist = self.cap_wrist.read()
        if not (ok1 and ok2):
            raise RuntimeError("camera read failed")
        # OpenCV is BGR, dataset stored RGB.
        frame_global = self.cv2.cvtColor(frame_global, self.cv2.COLOR_BGR2RGB)
        frame_wrist = self.cv2.cvtColor(frame_wrist, self.cv2.COLOR_BGR2RGB)
        return frame_global, frame_wrist

    def close(self) -> None:
        self.cap_global.release()
        self.cap_wrist.release()


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

    def __init__(self, ip: str):
        try:
            import rtde_control
            import rtde_receive
        except ImportError as e:
            raise ImportError(
                "ur_rtde not installed. `pip install ur_rtde` and ensure UR controller is reachable."
            ) from e

        self.ctrl = rtde_control.RTDEControlInterface(ip)
        self.recv = rtde_receive.RTDEReceiveInterface(ip)
        # === HARDWARE_TODO: gripper SDK (Robotiq, Schunk, vacuum, etc.)
        # Example for Robotiq via socket: implement a simple class with
        #   .read_position() -> [0,1] and .write_position(value)
        self.gripper = None  # plug your gripper driver here

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
        self.ctrl.servoStop()
        self.ctrl.stopScript()


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

    cams = Cameras()
    robot = None if args.dry_run else URRobot(args.robot_ip)

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
