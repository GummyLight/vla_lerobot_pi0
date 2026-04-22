"""
Mode 1 — URScript + Robotiq 2F-58 + 2× Intel D435i

Sends URScript programs to the UR7e robot and simultaneously records:
  • Robot joint positions, velocities, TCP pose
  • Robotiq gripper position
  • Color frames from two D435i cameras

Data is saved in LeRobot v2.0 format (parquet + mp4).

Usage:
    python collect_urscript.py --config configs/urscript_config.yaml \\
                               --dataset_name my_dataset \\
                               --task "pick and place"
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from utils.camera_interface import MultiCamera
from utils.lerobot_writer import LeRobotWriter
from utils.robot_interface import UR7eInterface
from utils.robotiq_interface import RobotiqGripper


# ------------------------------------------------------------------
# State / action space definitions
# ------------------------------------------------------------------
STATE_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "gripper",
]
ACTION_NAMES = STATE_NAMES  # action = next joint + gripper target


def parse_args():
    p = argparse.ArgumentParser(description="URScript data collection for LeRobot")
    p.add_argument("--config", default="configs/urscript_config.yaml")
    p.add_argument("--output_dir", default="datasets")
    p.add_argument("--dataset_name", default="ur7e_urscript_demo")
    p.add_argument("--task", default="manipulation", help="Natural-language task description")
    p.add_argument("--urscript_file", default=None,
                   help="Path to a .script file to send at episode start. "
                        "If omitted you can type/paste URScript interactively.")
    p.add_argument("--fps", type=int, default=None, help="Override config fps")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Data collector
# ------------------------------------------------------------------

class URScriptCollector:
    def __init__(self, cfg: dict, writer: LeRobotWriter):
        self.cfg = cfg
        self.writer = writer

        self.robot = UR7eInterface(
            host=cfg["robot"]["host"],
            frequency=cfg["robot"].get("frequency", 500.0),
        )
        self.gripper = RobotiqGripper(
            host=cfg["robot"]["host"],
            port=cfg["gripper"].get("port", 63352),
        )
        cam_cfgs = cfg["cameras"]
        self.cameras = MultiCamera(cam_cfgs)
        self._cam_keys = [c["name"] for c in cam_cfgs]

    def connect(self):
        self.robot.connect()
        self.gripper.connect()
        self.cameras.connect()
        print("\n[Collector] All devices ready.\n")

    def disconnect(self):
        try:
            self.robot.stop()
        except Exception:
            pass
        self.robot.disconnect()
        self.gripper.disconnect()
        self.cameras.disconnect()

    def run_episode(self, task: str, urscript: str | None) -> bool:
        """
        Record one episode. Returns True to continue, False to quit.
        """
        print(f"\n{'='*60}")
        print(f"  Episode {self.writer.episode_index}  |  task: {task}")
        print(f"{'='*60}")

        # Ask for URScript if not supplied
        script_to_send = urscript
        if script_to_send is None:
            print("Enter URScript (finish with a line containing only 'END'), or 'skip' to record without sending:")
            lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                if line.strip().lower() == "skip":
                    lines = []
                    break
                lines.append(line)
            script_to_send = "\n".join(lines) if lines else None

        input("\nPress Enter to START recording (Ctrl+C to stop episode)...")

        fps = self.writer.fps
        period = 1.0 / fps
        self.writer.start_episode(task)

        if script_to_send:
            print("[Robot] Sending URScript...")
            self.robot.send_urscript(script_to_send)

        print(f"[Collector] Recording at {fps} fps. Press Ctrl+C to end episode.\n")
        start = time.time()
        n_frames = 0
        try:
            while True:
                t0 = time.time()

                robot_state = self.robot.get_state()
                gripper_pos = self.gripper.get_position()
                images = self.cameras.get_latest_frames()

                state = np.concatenate([
                    robot_state["joint_positions"],   # 6
                    [gripper_pos],                    # 1
                ]).astype(np.float32)

                # Placeholder action (will be shifted in end_episode)
                action = state.copy()

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

        print(f"\n[Collector] Episode captured: {n_frames} frames ({n_frames / fps:.1f}s)")

        # Mark the last frame as terminal
        if self.writer._rows:
            self.writer._rows[-1]["next.done"] = True

        choice = input("Save this episode? [Y/n/q] ").strip().lower()
        if choice == "q":
            self.writer.end_episode(discard=True)
            return False
        discard = choice == "n"
        self.writer.end_episode(discard=discard)
        return True

    def run(self, task: str, urscript_file: str | None):
        urscript = None
        if urscript_file:
            with open(urscript_file) as f:
                urscript = f.read()
            print(f"[Collector] Loaded URScript from {urscript_file}")

        try:
            while True:
                print("\nOptions:  [s] Start new episode  |  [q] Quit")
                cmd = input(">> ").strip().lower()
                if cmd == "q":
                    break
                if cmd in ("s", ""):
                    keep_going = self.run_episode(task, urscript)
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
    dataset_name = args.dataset_name or cfg["collection"].get("dataset_name", "ur7e_urscript_demo")

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
        action_is_commanded=False,  # actions derived by state-shift in end_episode
    )

    collector = URScriptCollector(cfg, writer)
    collector.connect()
    collector.run(task=args.task, urscript_file=args.urscript_file)


if __name__ == "__main__":
    main()
