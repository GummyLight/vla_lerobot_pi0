#!/usr/bin/env python3
"""
Help dial in ``teleoperation.safety.joint_limits`` for the Pika gripper.

Workflow
--------
1.  Start UR in Remote Control + ON + START.
2.  Run this script:
        python tools/probe_joint_limits.py
3.  The script puts the arm in **freedrive** mode and prints the live
    joint angles (rad) at 5 Hz.
4.  Slowly move the arm by hand to every extreme that you DO want to
    use during teleop. Press [Enter] to save the current pose as a
    "safe" sample. Repeat for many poses (10-30 samples).
5.  Press 'q' + Enter to stop. The script prints a min/max envelope
    over all samples — copy that into ``configs/pika_config.yaml`` as
    the ``joint_limits`` entry.

Notes
-----
- UR freedrive lets you push the arm by hand. Don't push so hard that
  you trigger a Protective Stop — the script will exit cleanly if it
  detects RTDE control loss.
- Don't move the arm to dangerous poses (gripper-on-arm collision)
  while sampling — those are exactly the poses you want EXCLUDED from
  the envelope.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from utils.robot_interface import UR7eInterface  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(_HERE / "configs" / "pika_config.yaml"))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    robot = UR7eInterface(host=cfg["robot"]["host"],
                          frequency=cfg["robot"].get("frequency", 500.0))
    robot.connect(use_control=True)
    print("[probe] Entering freedrive mode — push the arm by hand.")
    print("[probe] Each [Enter] saves the current pose as a 'safe' sample.")
    print("[probe] Type 'q' + Enter to finish and print the envelope.\n")
    robot.freedrive_mode(True)

    samples: list[np.ndarray] = []
    try:
        while True:
            q = robot.get_state()["joint_positions"]
            print(
                f"\rq = [{q[0]:+.2f}, {q[1]:+.2f}, {q[2]:+.2f}, "
                f"{q[3]:+.2f}, {q[4]:+.2f}, {q[5]:+.2f}]   "
                f"({len(samples)} samples)",
                end="", flush=True,
            )
            time.sleep(0.2)

            # Non-blocking-ish stdin read every loop — but actually we want
            # to block on Enter, so use a side thread? Simpler: just sleep
            # and watch stdin via select.
            import select
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                line = sys.stdin.readline().strip()
                if line.lower() == "q":
                    break
                samples.append(q.copy())
                print(f"\n[probe] Saved sample #{len(samples)}")
    finally:
        robot.freedrive_mode(False)
        robot.disconnect()

    if not samples:
        print("\n[probe] No samples collected.")
        return

    arr = np.stack(samples)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)

    print("\n\n[probe] Joint envelope from", len(samples), "samples:")
    margin = 0.10  # rad — pad envelope so live teleop has wiggle room
    print(f"\n  Suggested joint_limits (with ±{margin:.2f} rad padding):")
    print("\n  joint_limits:")
    for i in range(6):
        lo_pad = float(lo[i] - margin)
        hi_pad = float(hi[i] + margin)
        print(f"    - [{lo_pad:+.2f}, {hi_pad:+.2f}]   # q[{i}]")
    print("\n  Copy those 6 lines into the safety: block of pika_config.yaml.")


if __name__ == "__main__":
    main()
