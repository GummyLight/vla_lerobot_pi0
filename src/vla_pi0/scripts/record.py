"""Record demonstrations on a UR7e + Robotiq + 2× D435i rig.

Mirror of upstream `lerobot-record` for our hardware: instead of a bespoke
`URScriptCollector` that owns its own RTDE / RealSense plumbing, this
script asks a lerobot-style ``Robot`` (`UR7eFollower`) for observations
and writes them through ``LeRobotWriter`` — exactly the same flow that
`lerobot-record --robot.type=so100_follower` follows for SO-Arm rigs.

The actual motion that produces the demonstration still has to come from
somewhere; this script supports two sources:

  - ``--source urscript --urscript path/to/foo.script`` — replays a
    PolyScope URScript via the realtime interface (port 30003), like
    legacy `collect/collect_urscript.py`.
  - ``--source freedrive`` — enables UR teach mode and lets you guide the
    arm by hand. End the episode with Ctrl+C.

Use cases other than these (Pika teleoperation, leader-follower, etc.)
should subclass `Recorder` or just call into `UR7eFollower` from your own
loop — the Robot contract is intentionally narrow so any teleop layer can
plug in.

Usage:
    python -m vla_pi0.scripts.record \\
        --dataset-name my_demo \\
        --task "open the 3D printer" \\
        --source urscript \\
        --urscript "collect/urscripts/open the 3D printer.script" \\
        --robot.ip 192.168.1.100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect.utils.lerobot_writer import LeRobotWriter  # noqa: E402

from vla_pi0.robots.ur7e_follower import UR7eFollower, UR7eFollowerConfig  # noqa: E402
from vla_pi0.robots.ur7e_follower.config_ur7e_follower import CameraSpec  # noqa: E402


STATE_NAMES = (
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "gripper",
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="datasets")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--task", required=True,
                    help="Natural-language task description; saved verbatim into "
                         "meta/tasks.parquet so the trained policy can be "
                         "conditioned on it.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--source",
                    choices=["urscript", "freedrive"],
                    default="freedrive",
                    help="How motion is produced during recording. 'urscript' "
                         "replays a .script via the realtime interface; "
                         "'freedrive' uses UR teach mode (manual guiding).")
    ap.add_argument("--urscript", type=Path, default=None,
                    help="Required when --source=urscript: path to a .script "
                         "exported from PolyScope.")

    ap.add_argument("--robot.ip", dest="robot_ip", default="192.168.1.100")
    ap.add_argument("--robot.gripper-port", dest="robot_gripper_port",
                    type=int, default=63352)
    ap.add_argument("--robot.cameras.cam-global.serial",
                    dest="robot_cam_global_serial", default=None)
    ap.add_argument("--robot.cameras.cam-wrist.serial",
                    dest="robot_cam_wrist_serial", default=None)

    args = ap.parse_args()
    if args.source == "urscript" and not args.urscript:
        ap.error("--source=urscript requires --urscript path/to/foo.script")
    return args


def _make_robot(args: argparse.Namespace) -> UR7eFollower:
    cfg = UR7eFollowerConfig(
        ip=args.robot_ip,
        gripper_port=args.robot_gripper_port,
        cameras=[
            CameraSpec(name="cam_global", serial=args.robot_cam_global_serial),
            CameraSpec(name="cam_wrist", serial=args.robot_cam_wrist_serial),
        ],
    )
    return UR7eFollower(cfg)


def _record_episode(
    robot: UR7eFollower, writer: LeRobotWriter, task: str, fps: int,
    ur_kick: callable | None,
) -> int:
    """Drive one episode. Returns frame count actually written."""
    print(f"\n=== Episode {writer.episode_index} | task={task!r} ===")
    input("Press Enter to START recording (Ctrl+C to stop episode)...")
    writer.start_episode(task)

    if ur_kick is not None:
        ur_kick()

    period = 1.0 / fps
    start = time.time()
    n = 0
    try:
        while True:
            t0 = time.time()
            obs = robot.get_observation()

            state = np.array(
                [obs[f"{j}.pos"] for j in robot.config.joint_names]
                + [obs["gripper.pos"]],
                dtype=np.float32,
            )
            # action_is_commanded=False below tells the writer to derive
            # actions by shifting state by one timestep at end_episode().
            images = {c.name: obs[c.name] for c in robot.config.cameras}
            writer.add_frame(state=state, action=state.copy(), images=images)
            n += 1

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass

    print(f"  captured {n} frames ({n / fps:.1f}s)")
    if writer._rows:
        writer._rows[-1]["next.done"] = True

    choice = input("Save this episode? [Y/n/q] ").strip().lower()
    if choice == "q":
        writer.end_episode(discard=True)
        return -1
    writer.end_episode(discard=(choice == "n"))
    return n


def main() -> int:
    args = _parse_args()
    robot = _make_robot(args)
    robot.connect()

    cam_keys = [c.name for c in robot.config.cameras]
    img_size = (robot.config.cameras[0].height, robot.config.cameras[0].width)
    writer = LeRobotWriter(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        fps=args.fps,
        camera_keys=cam_keys,
        state_dim=len(STATE_NAMES),
        action_dim=len(STATE_NAMES),
        state_names=list(STATE_NAMES),
        action_names=list(STATE_NAMES),
        robot_type="ur7e",
        image_size=img_size,
        action_is_commanded=False,
    )

    ur_kick = None
    if args.source == "urscript":
        script = args.urscript.read_text(encoding="utf-8")

        def ur_kick():  # noqa: F811
            print(f"[record] replaying URScript ({len(script)} chars)")
            robot._robot.play_program(script)  # internal, fine here
    elif args.source == "freedrive":
        def ur_kick():
            print("[record] enabling freedrive teach mode")
            robot._robot.freedrive_mode(True)

    try:
        while True:
            print("\nOptions:  [s] start episode  |  [q] quit")
            cmd = input(">> ").strip().lower()
            if cmd == "q":
                break
            if cmd in ("", "s"):
                rc = _record_episode(robot, writer, args.task, args.fps, ur_kick)
                if rc < 0:
                    break
    finally:
        if args.source == "freedrive":
            try:
                robot._robot.freedrive_mode(False)
            except Exception:
                pass
        writer.finalize()
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
