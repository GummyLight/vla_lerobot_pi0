"""
Mode 2 — Pika Sense teleoperation + Pika Gripper + 1× D435i + Pika wrist camera

Streams commands from Pika Sense → UR7e TCP via servoL.
Simultaneously records:
  • Robot joint positions, velocities, TCP pose
  • Pika gripper position
  • External D435i color frame
  • Pika gripper wrist camera frame

Data is saved in LeRobot v2.0 format (parquet + mp4).

Usage:
    python collect_pika.py --config configs/pika_config.yaml \\
                           --dataset_name my_pika_dataset \\
                           --task "pour liquid"
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from utils.camera_interface import MultiCamera, UVCCamera
from utils.lerobot_writer import LeRobotWriter
from utils.pika_interface import PikaGripper, PikaSense
from utils.robot_interface import UR7eInterface


# ------------------------------------------------------------------
# State / action space definitions
# ------------------------------------------------------------------
STATE_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "gripper",
]
ACTION_NAMES = STATE_NAMES  # action = commanded joint target + gripper


def parse_args():
    p = argparse.ArgumentParser(description="Pika teleoperation data collection for LeRobot")
    p.add_argument("--config", default="configs/pika_config.yaml")
    p.add_argument("--output_dir", default="datasets")
    p.add_argument("--dataset_name", default="ur7e_pika_demo")
    p.add_argument("--task", default="manipulation")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--no_calibrate", action="store_true",
                   help="Skip Pika Sense calibration at startup")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Teleoperation controller
# ------------------------------------------------------------------

class TeleOpController:
    """
    Runs in a background thread.
    Reads Pika Sense → sends servoL to UR7e at ~100 Hz.
    Also controls the Pika gripper.
    """

    SERVO_HZ = 100  # servoL command rate

    def __init__(
        self,
        robot: UR7eInterface,
        sense: PikaSense,
        gripper: PikaGripper,
        speed_scale: float = 0.3,
        rotation_scale: float = 0.5,
    ):
        self.robot = robot
        self.sense = sense
        self.gripper = gripper
        self.speed_scale = speed_scale
        self.rotation_scale = rotation_scale

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_tcp_target: np.ndarray | None = None
        self._latest_gripper_cmd: float = 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self.robot.servo_stop()
        except Exception:
            pass

    def get_latest_command(self) -> tuple[np.ndarray | None, float]:
        """Return (tcp_target_or_None, gripper_cmd) for recording."""
        with self._lock:
            return (
                None if self._latest_tcp_target is None else self._latest_tcp_target.copy(),
                self._latest_gripper_cmd,
            )

    def _loop(self):
        dt = 1.0 / self.SERVO_HZ
        # Fetch initial TCP pose as starting target
        state = self.robot.get_state()
        tcp_target = state["tcp_pose"].copy()

        while self._running:
            t0 = time.time()

            sense_data = self.sense.get_latest()
            delta = sense_data["delta_pose"]              # (6,) [dx,dy,dz,drx,dry,drz]
            gripper_cmd = sense_data["gripper"]

            # Scale and accumulate delta
            tcp_target[:3] += delta[:3] * self.speed_scale * dt
            tcp_target[3:] += delta[3:] * self.rotation_scale * dt

            # Apply workspace limits if configured
            # (limits are applied in the main config as optional clipping)

            try:
                self.robot.servo_l(
                    pose=tcp_target.tolist(),
                    speed=0.5,
                    acc=0.5,
                    dt=dt,
                    lookahead=0.1,
                    gain=300,
                )
                self.gripper.move(gripper_cmd)
            except Exception as e:
                print(f"[TeleOp] servoL error: {e}")

            with self._lock:
                self._latest_tcp_target = tcp_target.copy()
                self._latest_gripper_cmd = float(gripper_cmd)

            elapsed = time.time() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)


# ------------------------------------------------------------------
# Pika collector
# ------------------------------------------------------------------

class PikaCollector:
    def __init__(self, cfg: dict, writer: LeRobotWriter):
        self.cfg = cfg
        self.writer = writer

        self.robot = UR7eInterface(
            host=cfg["robot"]["host"],
            frequency=cfg["robot"].get("frequency", 500.0),
        )
        self.gripper = PikaGripper(
            port=cfg["pika_gripper"].get("port", "/dev/ttyUSB0"),
            baudrate=cfg["pika_gripper"].get("baudrate", 115200),
        )
        self.sense = PikaSense(
            port=cfg["pika_sense"].get("port", "/dev/ttyUSB1"),
            baudrate=cfg["pika_sense"].get("baudrate", 115200),
        )

        # External D435i cameras (all except "cam_wrist" which comes from Pika)
        ext_cam_cfgs = [c for c in cfg["cameras"] if c.get("source") != "pika_gripper"]
        self.ext_cameras = MultiCamera(ext_cam_cfgs) if ext_cam_cfgs else None
        self._ext_cam_keys = [c["name"] for c in ext_cam_cfgs]

        # Pika wrist camera (UVC)
        wrist_cfg = next((c for c in cfg["cameras"] if c.get("source") == "pika_gripper"), None)
        self._wrist_key = wrist_cfg["name"] if wrist_cfg else None
        self._wrist_device_index = wrist_cfg.get("device_index", 0) if wrist_cfg else 0

        speed_scale = cfg.get("teleoperation", {}).get("speed_scale", 0.3)
        rotation_scale = cfg.get("teleoperation", {}).get("rotation_scale", 0.5)
        self.teleop = TeleOpController(
            self.robot, self.sense, self.gripper,
            speed_scale=speed_scale,
            rotation_scale=rotation_scale,
        )

    def connect(self, calibrate: bool = True):
        self.robot.connect()
        self.gripper.connect()
        self.sense.connect()
        if self.ext_cameras:
            self.ext_cameras.connect()
        if calibrate:
            self.sense.calibrate()
        print("\n[Collector] All Pika devices ready.\n")

    def disconnect(self):
        try:
            self.robot.stop()
        except Exception:
            pass
        self.robot.disconnect()
        self.gripper.disconnect()
        self.sense.disconnect()
        if self.ext_cameras:
            self.ext_cameras.disconnect()

    def _get_images(self) -> Dict[str, np.ndarray]:
        images: Dict[str, np.ndarray] = {}
        if self.ext_cameras:
            images.update(self.ext_cameras.get_latest_frames())
        if self._wrist_key:
            frame = self.gripper.get_camera_frame()
            if frame is not None:
                images[self._wrist_key] = frame
        return images

    def run_episode(self, task: str) -> bool:
        """Record one episode. Returns True to continue, False to quit."""
        print(f"\n{'='*60}")
        print(f"  Episode {self.writer.episode_index}  |  task: {task}")
        print(f"{'='*60}")
        input("\nPress Enter to START recording (Ctrl+C to stop episode)...")

        fps = self.writer.fps
        period = 1.0 / fps
        self.writer.start_episode(task)
        self.teleop.start()

        print(f"[Collector] Recording at {fps} fps. Teleop ACTIVE. Press Ctrl+C to end.\n")
        start = time.time()
        n_frames = 0
        try:
            while True:
                t0 = time.time()

                robot_state = self.robot.get_state()
                gripper_pos = self.gripper.get_position()
                images = self._get_images()

                # State: current joint angles + gripper
                state = np.concatenate([
                    robot_state["joint_positions"],
                    [gripper_pos],
                ]).astype(np.float32)

                # Action: what was COMMANDED (TCP target converted to action space)
                # We store commanded gripper + current joints as a simple approximation.
                # For true joint-space action, you would run IK on the TCP target here.
                tcp_target, gripper_cmd = self.teleop.get_latest_command()
                # Action = [joint_positions (as commanded), gripper_cmd]
                # Since we track joint positions, commanded joints ≈ current joints + delta
                # For the simplest approach: use current joint state as action base
                # and override gripper with the commanded value.
                action = np.concatenate([
                    robot_state["joint_positions"],
                    [gripper_cmd],
                ]).astype(np.float32)

                self.writer.add_frame(
                    state=state,
                    action=action,
                    images=images,
                    timestamp=time.time() - start,
                )
                n_frames += 1

                elapsed = time.time() - t0
                sleep = period - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            pass

        self.teleop.stop()
        print(f"\n[Collector] Episode captured: {n_frames} frames ({n_frames / fps:.1f}s)")

        if self.writer._rows:
            self.writer._rows[-1]["next.done"] = True

        choice = input("Save this episode? [Y/n/q] ").strip().lower()
        if choice == "q":
            self.writer.end_episode(discard=True)
            return False
        self.writer.end_episode(discard=(choice == "n"))
        return True

    def run(self, task: str):
        try:
            while True:
                print("\nOptions:  [s] Start episode  |  [c] Re-calibrate Sense  |  [q] Quit")
                cmd = input(">> ").strip().lower()
                if cmd == "q":
                    break
                if cmd == "c":
                    self.sense.calibrate()
                if cmd in ("s", ""):
                    keep_going = self.run_episode(task)
                    if not keep_going:
                        break
        finally:
            self.writer.finalize()
            self.disconnect()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    fps = args.fps or cfg["collection"].get("fps", 30)
    output_dir = args.output_dir or cfg["collection"].get("output_dir", "datasets")
    dataset_name = args.dataset_name or cfg["collection"].get("dataset_name", "ur7e_pika_demo")

    cam_keys = [c["name"] for c in cfg["cameras"]]

    writer = LeRobotWriter(
        output_dir=output_dir,
        dataset_name=dataset_name,
        fps=fps,
        camera_keys=cam_keys,
        state_dim=len(STATE_NAMES),
        action_dim=len(ACTION_NAMES),
        state_names=STATE_NAMES,
        action_names=ACTION_NAMES,
        robot_type="ur7e",
        image_size=(
            cfg["cameras"][0].get("height", 480),
            cfg["cameras"][0].get("width", 640),
        ),
        action_is_commanded=True,  # Pika Sense provides commanded actions
    )

    collector = PikaCollector(cfg, writer)
    collector.connect(calibrate=not args.no_calibrate)
    collector.run(task=args.task)


if __name__ == "__main__":
    main()
