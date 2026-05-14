"""PikaSense -> Robotiq gripper-only test.

This script never connects to UR RTDE and never sends arm motion commands. It
only uses the UR controller IP as the host for the Robotiq URCap socket
(`63352` by default).

Examples:
    python3 collect/test_pikasense_robotiq_gripper.py \
        --config collect/configs/pika_robotiq_config.yaml

    python3 collect/test_pikasense_robotiq_gripper.py \
        --config collect/configs/pika_robotiq_config.yaml \
        --robot-ip 169.254.26.10 --sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


_HERE = Path(__file__).resolve().parent
_PIKA_SDK = _HERE / "pika_sdk"
if _PIKA_SDK.exists() and str(_PIKA_SDK) not in sys.path:
    sys.path.insert(0, str(_PIKA_SDK))

from utils.gripper_adapters import GripperMapping, RobotiqGripperAdapter  # noqa: E402
from utils.pika_interface import PikaSense, detect_pika_ports             # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Test Robotiq open/close from PikaSense, without moving UR7e.")
    p.add_argument("--config", default="collect/configs/pika_robotiq_config.yaml")
    p.add_argument("--robot-ip", default=None,
                   help="Override robot.host from config. Robotiq socket host.")
    p.add_argument("--sense-port", default=None,
                   help="Override pika_sense.port. Empty config auto-detects.")
    p.add_argument("--rate", type=float, default=30.0,
                   help="PikaSense encoder polling rate in Hz.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Stop after N seconds. 0 means run until Ctrl+C.")
    p.add_argument("--quiet", action="store_true",
                   help="Print less often.")
    p.add_argument("--sweep", action="store_true",
                   help="Do not use PikaSense; cycle Robotiq closed/open.")
    p.add_argument("--sweep-period", type=float, default=2.0,
                   help="Seconds per closed/open command in --sweep mode.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_robotiq(cfg: dict) -> RobotiqGripperAdapter:
    robotiq_cfg = cfg.get("robotiq_gripper") or cfg.get("gripper") or {}
    return RobotiqGripperAdapter(
        host=cfg["robot"]["host"],
        port=int(robotiq_cfg.get("port", 63352)),
        mapping=GripperMapping.from_config(cfg.get("gripper_mapping")),
        force=int(robotiq_cfg.get("force", 150)),
        speed_min=int(robotiq_cfg.get("speed_min", 80)),
        speed_max=int(robotiq_cfg.get("speed_max", 255)),
        max_norm_speed_per_s=float(robotiq_cfg.get("max_norm_speed_per_s", 2.0)),
    )


def run_sweep(gripper: RobotiqGripperAdapter, period_s: float,
              duration_s: float, quiet: bool) -> None:
    period_s = max(0.2, float(period_s))
    deadline = None if duration_s <= 0 else time.time() + duration_s
    closed = True
    while deadline is None or time.time() < deadline:
        pos = 1.0 if closed else 0.0
        gripper.set_replay_position(pos)
        if not quiet:
            print(f"[sweep] Robotiq target={pos:.1f} "
                  f"({'closed' if closed else 'open'})")
        closed = not closed
        time.sleep(period_s)


def run_pikasense(cfg: dict, gripper: RobotiqGripperAdapter,
                  sense_port: str | None, rate_hz: float,
                  duration_s: float, quiet: bool) -> None:
    sense_cfg = cfg.get("pika_sense", {})
    port = sense_port if sense_port is not None else (sense_cfg.get("port") or "")
    if not port:
        port, _ = detect_pika_ports(None, None)
        print(f"[test] Auto-detected Pika Sense port — sense={port}")

    sense = PikaSense(
        port=port,
        tracker_device=sense_cfg.get("tracker_device", "T20"),
        tracker_config=sense_cfg.get("tracker_config"),
        tracker_lh_config=sense_cfg.get("tracker_lh_config"),
    )
    sense.connect()
    period = 1.0 / max(1.0, float(rate_hz))
    deadline = None if duration_s <= 0 else time.time() + duration_s
    next_print = 0.0
    try:
        while deadline is None or time.time() < deadline:
            tick = time.time()
            rad = sense.get_encoder_rad()
            cmd = gripper.command_from_pika_encoder(rad, period)
            if not quiet and tick >= next_print:
                actual = gripper.read_position()
                print(f"[test] pika_rad={rad:.3f} -> robotiq_cmd={cmd:.3f} "
                      f"actual={actual:.3f}")
                next_print = tick + 0.5
            sleep_s = period - (time.time() - tick)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        sense.disconnect()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.robot_ip:
        cfg.setdefault("robot", {})["host"] = args.robot_ip

    gripper = make_robotiq(cfg)
    print(f"[test] Connecting Robotiq @ {cfg['robot']['host']}:"
          f"{(cfg.get('robotiq_gripper') or {}).get('port', 63352)}")
    gripper.connect()
    try:
        if args.sweep:
            run_sweep(gripper, args.sweep_period, args.duration, args.quiet)
        else:
            run_pikasense(
                cfg=cfg,
                gripper=gripper,
                sense_port=args.sense_port,
                rate_hz=args.rate,
                duration_s=args.duration,
                quiet=args.quiet,
            )
    except KeyboardInterrupt:
        pass
    finally:
        gripper.disconnect()
        print("[test] Done. UR7e arm was not commanded.")


if __name__ == "__main__":
    main()
