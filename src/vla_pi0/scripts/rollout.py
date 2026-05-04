"""Closed-loop policy inference on a UR7e + Robotiq + 2× D435i rig.

Mirror of upstream `lerobot-rollout` for our hardware: a single CLI that
loads a trained pi0/SmolVLA policy and drives a real robot through the
``Robot.get_observation() → policy.select_action() → Robot.send_action()``
contract.

Differences from the legacy `deploy/run_pi0_robot.py`:
- No bespoke `Cameras` / `URRobot` classes — everything goes through the
  ``UR7eFollower`` lerobot ``Robot`` subclass in `vla_pi0.robots`.
- Hardware config is a dataclass (``UR7eFollowerConfig``); CLI flags map to
  its fields exactly the way `lerobot-rollout --robot.ip=...` does.
- The control loop is a thin ``Strategy``-style function — it never touches
  the cameras or RTDE directly, which makes swapping in a different
  arm/teleop trivial.

Usage:
    python -m vla_pi0.scripts.rollout \\
        --policy-path outputs/train/pi0_3d_printer_lora/checkpoints/last/pretrained_model \\
        --task "open the 3D printer" \\
        --robot.ip 192.168.1.100 \\
        --robot.cameras.cam_global.serial 123456789 \\
        --robot.cameras.cam_wrist.serial 987654321 \\
        --max-seconds 15
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))  # so eval_pi0 imports resolve

# Match the train + legacy eval scripts' HF cache so we hit the cached pi0 base.
if sys.platform == "win32":
    os.environ.setdefault("HF_HOME", r"D:\.hfcache")
else:
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))

from eval_pi0 import load_policy_with_lora, load_processors  # noqa: E402

from vla_pi0.robots.ur7e_follower import (  # noqa: E402
    UR7eFollower,
    UR7eFollowerConfig,
)
from vla_pi0.robots.ur7e_follower.config_ur7e_follower import CameraSpec  # noqa: E402


@dataclass
class RolloutArgs:
    policy_path: Path
    task: str
    max_seconds: float
    device: str
    dry_run: bool
    loop: bool
    robot_cfg: UR7eFollowerConfig


def _parse_args() -> RolloutArgs:
    """Hand-rolled argparse-based CLI.

    We do NOT pull draccus in here even though upstream lerobot does — keeping
    this script importable in environments where lerobot's optional draccus
    extras aren't installed (e.g. CI smoke tests). The flag layout still
    mirrors `--robot.<field>` so the muscle-memory carries over.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", type=Path, required=True)
    ap.add_argument("--task", required=True,
                    help="Task string. MUST appear verbatim in the dataset's "
                         "tasks.parquet — pi0 was conditioned on these exact strings.")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the loop but DO NOT connect to the robot — observations "
                         "come from cameras only, joints are zero, actions printed.")
    ap.add_argument("--loop", action="store_true",
                    help="After each rollout, prompt for next task/duration.")

    # Robot.* — flat to keep argparse simple. Same names as the dataclass fields.
    ap.add_argument("--robot.ip", dest="robot_ip", default="192.168.1.100")
    ap.add_argument("--robot.gripper-port", dest="robot_gripper_port", type=int, default=63352)
    ap.add_argument("--robot.control-hz", dest="robot_control_hz", type=float, default=30.0)
    ap.add_argument("--robot.max-joint-delta-rad", dest="robot_max_joint_delta_rad",
                    type=float, default=0.10)
    ap.add_argument("--robot.clamp-mode", dest="robot_clamp_mode",
                    choices=["refuse", "clip"], default="refuse")
    ap.add_argument("--robot.gripper-threshold", dest="robot_gripper_threshold",
                    type=float, default=0.5)
    ap.add_argument("--robot.cameras.cam-global.serial",
                    dest="robot_cam_global_serial", default=None)
    ap.add_argument("--robot.cameras.cam-wrist.serial",
                    dest="robot_cam_wrist_serial", default=None)

    args = ap.parse_args()
    robot_cfg = UR7eFollowerConfig(
        ip=args.robot_ip,
        gripper_port=args.robot_gripper_port,
        control_hz=args.robot_control_hz,
        max_joint_delta_rad=args.robot_max_joint_delta_rad,
        clamp_mode=args.robot_clamp_mode,
        gripper_threshold=args.robot_gripper_threshold,
        cameras=[
            CameraSpec(name="cam_global", serial=args.robot_cam_global_serial),
            CameraSpec(name="cam_wrist", serial=args.robot_cam_wrist_serial),
        ],
    )
    return RolloutArgs(
        policy_path=args.policy_path,
        task=args.task,
        max_seconds=args.max_seconds,
        device=args.device,
        dry_run=args.dry_run,
        loop=args.loop,
        robot_cfg=robot_cfg,
    )


def _build_dataset_observation(
    obs: dict[str, Any], task: str, joint_names: tuple[str, ...], device: str
) -> dict[str, Any]:
    """Translate a Robot observation dict into the keys pi0 was trained on.

    UR7eFollower returns per-joint float scalars and per-camera HxWx3 uint8
    arrays; pi0 expects a 7-d `observation.state` and CHW float32 image
    tensors under `observation.images.<cam>`.
    """
    state = np.array(
        [obs[f"{j}.pos"] for j in joint_names] + [obs["gripper.pos"]],
        dtype=np.float32,
    )

    def to_chw_float01(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    out: dict[str, Any] = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [task],
    }
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            out[f"observation.images.{k}"] = to_chw_float01(v).unsqueeze(0).to(device)
    return out


def _action_tensor_to_dict(
    action: torch.Tensor, joint_names: tuple[str, ...]
) -> dict[str, Any]:
    """Inverse of _build_dataset_observation: 7-d tensor → per-joint dict."""
    arr = action.squeeze(0).detach().cpu().float().numpy()
    out: dict[str, Any] = {f"{j}.pos": float(arr[i]) for i, j in enumerate(joint_names)}
    out["gripper.pos"] = float(arr[6])
    return out


def run_rollout(
    policy, pre, post, robot: UR7eFollower | None, cfg: UR7eFollowerConfig,
    args: RolloutArgs, task: str, max_seconds: float,
) -> None:
    """Single rollout. Caller is responsible for connect/disconnect."""
    underlying = policy.base_model if hasattr(policy, "base_model") else policy
    if hasattr(underlying, "reset"):
        underlying.reset()

    period = 1.0 / cfg.control_hz
    t0 = time.perf_counter()
    n_steps = 0

    print(
        f"starting rollout: task={task!r}  max={max_seconds}s  "
        f"hz={cfg.control_hz:.0f}  dry_run={robot is None}"
    )
    try:
        while time.perf_counter() - t0 < max_seconds:
            tick = time.perf_counter()

            if robot is not None:
                obs_raw = robot.get_observation()
            else:
                # Dry-run: zeros for joints, but still pull camera frames so
                # we exercise the same decode/encode path.
                obs_raw = {f"{j}.pos": 0.0 for j in cfg.joint_names}
                obs_raw["gripper.pos"] = 0.0
                # In a true dry run without robot we still want camera frames;
                # the cleanest way is to spin up the robot's camera-only mode,
                # but that's a future cleanup. For now: fabricate zeros.
                for cam in cfg.cameras:
                    obs_raw[cam.name] = np.zeros(
                        (cam.height, cam.width, 3), dtype=np.uint8
                    )

            batch = _build_dataset_observation(obs_raw, task, cfg.joint_names, args.device)
            batch = pre(batch)
            with torch.no_grad():
                action = policy.select_action(batch)
            action = post(action)
            action_dict = _action_tensor_to_dict(action, cfg.joint_names)

            if robot is not None:
                sent = robot.send_action(action_dict)
                if n_steps % 30 == 0:
                    qs = [sent[f"{j}.pos"] for j in cfg.joint_names]
                    print(
                        f"step {n_steps:4d}  q={np.round(qs, 3)}  "
                        f"gripper={sent['gripper.pos']:.2f}"
                    )
            else:
                if n_steps % 30 == 0:
                    qs = [action_dict[f"{j}.pos"] for j in cfg.joint_names]
                    print(
                        f"step {n_steps:4d}  q={np.round(qs, 3)}  "
                        f"gripper={action_dict['gripper.pos']:.2f}  [dry-run]"
                    )

            n_steps += 1

            elapsed = time.perf_counter() - tick
            if elapsed < period:
                time.sleep(period - elapsed)
            elif elapsed > 1.5 * period:
                print(
                    f"⚠ step {n_steps}: control loop slow "
                    f"({elapsed * 1000:.0f}ms > {period * 1000:.0f}ms)"
                )
            actual_dt = max(time.perf_counter() - tick, period)
            if robot is not None:
                robot.note_loop_dt(actual_dt)
    finally:
        if robot is not None:
            robot.servo_stop()
        dur = time.perf_counter() - t0
        print(
            f"rollout done. {n_steps} steps in {dur:.1f}s "
            f"({n_steps / max(dur, 1e-3):.1f} Hz avg)"
        )


def _prompt_next(default_task: str, default_seconds: float) -> tuple[str, float] | None:
    print()
    print(
        f"--- next rollout (defaults: task={default_task!r}, "
        f"max_seconds={default_seconds}) ---"
    )
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
    args = _parse_args()

    print(f"loading policy from: {args.policy_path}")
    policy = load_policy_with_lora(args.policy_path, args.device)
    pre, post = load_processors(args.policy_path)

    robot: UR7eFollower | None = None
    if not args.dry_run:
        robot = UR7eFollower(args.robot_cfg)
        robot.connect()

    try:
        run_rollout(policy, pre, post, robot, args.robot_cfg, args, args.task, args.max_seconds)

        if args.loop:
            task, secs = args.task, args.max_seconds
            while True:
                nxt = _prompt_next(task, secs)
                if nxt is None:
                    break
                task, secs = nxt
                run_rollout(policy, pre, post, robot, args.robot_cfg, args, task, secs)
    finally:
        if robot is not None:
            robot.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
