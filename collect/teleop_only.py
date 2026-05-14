"""
Standalone Pika teleoperation — drives the UR7e + Pika gripper from the
Pika Sense / Vive tracker, **without** recording any data.

Use this to:
  * verify the teleop pipeline before starting a real data-collection session
  * tune PIKA_SCALE / PIKA_MAX_DELTA_M for your workspace
  * manually reposition the arm between episodes (engage trigger, drag the
    handle, release)

It uses exactly the same config file, vendored SDK, and ``PikaTeleopController``
as ``collect_pika.py`` — so the behavior you see here is identical to what
``collect_pika.py`` would do (minus the writer).

Usage
-----
    python teleop_only.py                              # uses configs/pika_config.yaml
    PIKA_SCALE=0.3 python teleop_only.py               # finer-grained
    python teleop_only.py --config /path/to/other.yaml

Operator flow
-------------
    1. Pull Pika Sense trigger          → [Teleop] >> ENGAGED
    2. Move handle / squeeze for grip   → arm + gripper follow
    3. Pull trigger again               → [Teleop] << RELEASED
    4. Ctrl+C                           → clean shutdown
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Make the vendored pika SDK importable without `pip install`.
_HERE = Path(__file__).resolve().parent
_PIKA_SDK = _HERE / "pika_sdk"
if _PIKA_SDK.exists() and str(_PIKA_SDK) not in sys.path:
    sys.path.insert(0, str(_PIKA_SDK))

# Reuse the exact same teleop logic that collect_pika.py uses.
from collect_pika import PikaTeleopController                          # noqa: E402
from utils.camera_interface import MultiCamera                          # noqa: E402
from utils.gripper_adapters import make_gripper_backend                   # noqa: E402
from utils.pika_interface import PikaSense, detect_pika_ports             # noqa: E402
from utils.preview import CameraPreviewer                               # noqa: E402
from utils.robot_interface import UR7eInterface                         # noqa: E402


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("teleop_only")


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone Pika teleop (no recording)")
    p.add_argument("--config", default="configs/pika_config.yaml")
    p.add_argument("--no_preview", action="store_true",
                   help="Disable the live RGB preview window.")
    p.add_argument("--record_path", default="",
                   help="If set, record a joint+gripper trajectory to this .npz path (Ctrl+C to stop).")
    p.add_argument("--record_hz", type=float, default=30.0,
                   help="Recording frequency (Hz) when --record_path is set.")
    p.add_argument("--replay_path", default="",
                   help="If set, replay a previously recorded .npz trajectory and exit.")
    p.add_argument("--replay_use", choices=["target_q", "actual_q"], default="target_q",
                   help="Which joint trajectory to replay from the file.")
    p.add_argument("--replay_hz", type=float, default=0.0,
                   help="If >0, override replay frequency (Hz). Otherwise uses recorded timestamps.")
    p.add_argument("--replay_speed", type=float, default=1.0,
                   help="Playback speed multiplier (e.g. 0.5 slower, 2.0 faster).")
    p.add_argument("--replay_after_record", action="store_true",
                   help="If set together with --record_path, replay immediately after recording.")
    p.add_argument("--robot-ip", default=None,
                   help="Override robot.host from the config.")
    p.add_argument("--sense-port", default=None,
                   help="Override pika_sense.port from the config.")
    p.add_argument("--gripper-backend", choices=["pika", "robotiq"], default=None,
                   help="Override gripper_mapping.type from config.")
    return p.parse_args()


def _np_str(v) -> "np.ndarray":
    import numpy as np
    return np.asarray(str(v), dtype=np.str_)


def _record_trajectory(robot: UR7eInterface, gripper,
                       teleop: PikaTeleopController, out_path: str,
                       record_hz: float, meta: dict) -> str:
    import numpy as np

    out_path = str(Path(out_path).expanduser().resolve())
    period = 1.0 / max(1e-6, float(record_hz))

    print(f"\n[teleop_only] Recording to: {out_path}")
    input("[teleop_only] Press Enter to START recording (Ctrl+C to stop)... ")

    t0 = time.perf_counter()
    ts = []
    q_actual = []
    q_target = []
    gripper_actual = []
    gripper_cmd = []

    n = 0
    try:
        while True:
            tick = time.perf_counter()
            state = robot.get_state()
            ts.append(tick - t0)
            q_actual.append(state["joint_positions"].astype(np.float32, copy=False))
            q_target.append(robot.get_target_q().astype(np.float32, copy=False))
            gripper_actual.append(np.float32(gripper.read_position()))
            gripper_cmd.append(np.float32(teleop.get_command_snapshot()["gripper_cmd"]))
            n += 1

            elapsed = time.perf_counter() - tick
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass

    if n <= 0:
        raise RuntimeError("No frames recorded.")

    np.savez_compressed(
        out_path,
        timestamps_s=np.asarray(ts, dtype=np.float64),
        q_actual=np.stack(q_actual, axis=0),
        q_target=np.stack(q_target, axis=0),
        gripper_actual=np.asarray(gripper_actual, dtype=np.float32),
        gripper_cmd=np.asarray(gripper_cmd, dtype=np.float32),
        meta_yaml=_np_str(yaml.safe_dump(meta, sort_keys=True, allow_unicode=True)),
    )
    dur = float(ts[-1]) if ts else 0.0
    print(f"[teleop_only] Recorded: {n} frames  ({dur:.2f}s)  hz≈{n / max(dur, 1e-6):.1f}")
    return out_path


def _replay_trajectory(robot: UR7eInterface, gripper,
                       npz_path: str, use: str,
                       replay_hz: float, speed: float,
                       lookahead_s: float, gain: float) -> None:
    import numpy as np

    npz_path = str(Path(npz_path).expanduser().resolve())
    data = np.load(npz_path, allow_pickle=False)

    if use == "target_q" and "q_target" in data:
        q_seq = data["q_target"].astype(np.float32, copy=False)
    else:
        q_seq = data["q_actual"].astype(np.float32, copy=False)

    if "gripper_cmd" in data:
        g_seq = data["gripper_cmd"].astype(np.float32, copy=False)
    elif "gripper_actual" in data:
        g_seq = data["gripper_actual"].astype(np.float32, copy=False)
    elif "gripper_rad_cmd" in data:
        g_seq = data["gripper_rad_cmd"].astype(np.float32, copy=False)
    elif "gripper_rad_actual" in data:
        g_seq = data["gripper_rad_actual"].astype(np.float32, copy=False)
    else:
        g_seq = np.zeros((q_seq.shape[0],), dtype=np.float32)

    ts = data["timestamps_s"].astype(np.float64, copy=False) if "timestamps_s" in data else None
    n = int(q_seq.shape[0])
    if n <= 0:
        raise RuntimeError("Empty trajectory.")

    speed = float(speed)
    if speed <= 0:
        speed = 1.0

    if float(replay_hz) > 0:
        period = 1.0 / float(replay_hz)
    elif ts is not None and ts.size >= 2:
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        period = float(np.median(dt)) / speed if dt.size else 1.0 / 50.0
    else:
        period = 1.0 / 50.0

    print(f"\n[teleop_only] Replaying: {npz_path}")
    print(f"[teleop_only] steps={n}  use={use}  period={period:.4f}s  speed={speed:.2f}x")
    input("[teleop_only] Press Enter to START replay (Ctrl+C to abort)... ")

    try:
        robot.move_j(q_seq[0].tolist(), speed=0.2, acc=0.2, asynchronous=False)
    except Exception:
        pass

    prev_loop_dt = period
    t0 = time.perf_counter()
    try:
        for i in range(n):
            tick = time.perf_counter()
            q = q_seq[i].tolist()
            g = float(g_seq[i]) if i < g_seq.shape[0] else 0.0

            robot.servo_j(q, dt=prev_loop_dt, lookahead=lookahead_s, gain=gain)
            gripper.set_replay_position(g)

            elapsed = time.perf_counter() - tick
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
            prev_loop_dt = max(time.perf_counter() - tick, period)
    except KeyboardInterrupt:
        print("\n[teleop_only] Replay aborted by user.")
    finally:
        try:
            robot.servo_stop()
        except Exception:
            pass
    print(f"[teleop_only] Replay done  ({time.perf_counter() - t0:.2f}s wall).")


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.robot_ip:
        cfg.setdefault("robot", {})["host"] = args.robot_ip
    if args.sense_port:
        cfg.setdefault("pika_sense", {})["port"] = args.sense_port

    sense_cfg = cfg.get("pika_sense", {})
    teleop_cfg = cfg.get("teleoperation", {})
    gripper_backend = (
        args.gripper_backend
        or (cfg.get("gripper_mapping") or {}).get("type")
        or "pika"
    ).lower()

    replay_only = bool(args.replay_path) and not bool(args.record_path)

    sense_port_pref = sense_cfg.get("port") or ""
    gripper_port_pref = (
        cfg.get("pika_gripper", {}).get("port") or ""
        if gripper_backend == "pika"
        else None
    )
    if ((not replay_only and not sense_port_pref)
            or (gripper_backend == "pika" and not gripper_port_pref)):
        sp, gp = detect_pika_ports(sense_port_pref or None,
                                   gripper_port_pref or None)
        sense_port_pref = sense_port_pref or sp
        if gripper_backend == "pika":
            gripper_port_pref = gripper_port_pref or gp
            print(f"[teleop_only] Auto-detected ports — sense={sense_port_pref}, "
                  f"gripper={gripper_port_pref}")
        else:
            print(f"[teleop_only] Auto-detected Pika Sense port — "
                  f"sense={sense_port_pref}")

    robot = UR7eInterface(host=cfg["robot"]["host"],
                          frequency=cfg["robot"].get("frequency", 500.0))
    sense = None
    if not replay_only:
        sense = PikaSense(
            port=sense_port_pref,
            tracker_device=sense_cfg.get("tracker_device", "T20"),
            tracker_config=sense_cfg.get("tracker_config"),
            tracker_lh_config=sense_cfg.get("tracker_lh_config"),
        )

    # Cameras — same wiring as collect_pika.py so you see exactly what the
    # collector would capture. Disabled with --no_preview.
    show_preview = not args.no_preview
    wrist_cam = next((c for c in cfg.get("cameras", [])
                      if c.get("type") == "pika_wrist"
                      or c.get("source") == "pika_gripper"), None)
    wrist_key = (
        wrist_cam["name"]
        if (wrist_cam and show_preview and gripper_backend == "pika")
        else None
    )

    gripper = make_gripper_backend(
        gripper_backend,
        cfg,
        wrist_cam=wrist_cam,
        show_preview=show_preview,
    )
    if gripper_backend == "pika" and gripper_port_pref:
        gripper.port = gripper_port_pref

    ext_cams = None
    ext_keys: list[str] = []
    if show_preview:
        ext_cfgs = [c for c in cfg.get("cameras", [])
                    if c.get("type") != "pika_wrist"
                    and c.get("source") != "pika_gripper"]
        if ext_cfgs:
            ext_cams = MultiCamera(ext_cfgs)
            ext_keys = [c["name"] for c in ext_cfgs]

    import os
    position_scale = float(os.environ.get(
        "PIKA_SCALE", teleop_cfg.get("position_scale", 1.0)))
    max_delta_m = float(os.environ.get(
        "PIKA_MAX_DELTA_M", teleop_cfg.get("max_delta_m", 1.0)))
    smoothing_cfg = teleop_cfg.get("smoothing", {}) or {}
    smoothing_alpha = float(os.environ.get(
        "PIKA_SMOOTHING_ALPHA",
        smoothing_cfg.get("pose_alpha", 1.0)))
    gripper_smoothing_alpha = float(os.environ.get(
        "PIKA_GRIPPER_SMOOTHING_ALPHA",
        smoothing_cfg.get("gripper_alpha", 1.0)))
    safety_cfg = teleop_cfg.get("safety", {}) or {}
    import numpy as np
    max_tilt_deg = safety_cfg.get("max_tilt_from_down_deg")
    max_tilt_rad = (None if max_tilt_deg is None
                    else float(max_tilt_deg) * np.pi / 180.0)

    teleop = None
    servo_lookahead_s = float(teleop_cfg.get("servo_lookahead_s", 0.2))
    servo_gain = float(teleop_cfg.get("servo_gain", 100.0))
    if not replay_only:
        teleop = PikaTeleopController(
            robot=robot,
            sense=sense,
            gripper=gripper,
            pika_to_arm=teleop_cfg.get(
                "pika_to_arm",
                [0.0, 0.0, 0.0, 1.703151, 1.539109, 1.728148],
            ),
            position_scale=position_scale,
            max_delta_m=max_delta_m,
            servo_hz=int(teleop_cfg.get("servo_hz", 50)),
            smoothing_alpha=smoothing_alpha,
            gripper_smoothing_alpha=gripper_smoothing_alpha,
            workspace_bounds=safety_cfg.get("workspace") or {},
            joint_limits=safety_cfg.get("joint_limits"),
            max_tilt_from_down_rad=max_tilt_rad,
            ik_mode=teleop_cfg.get("ik_mode", "ur_native_servol"),
            base_bias_min_radius_m=float(
                teleop_cfg.get("base_bias_min_radius_m", 0.05)),
            servo_lookahead_s=servo_lookahead_s,
            servo_gain=servo_gain,
            max_lin_vel_m_s=float(teleop_cfg.get("max_lin_vel_m_s", 0.30)),
            max_ang_vel_rad_s=float(teleop_cfg.get("max_ang_vel_rad_s", 1.50)),
            max_joint_vel_rad_s=float(teleop_cfg.get("max_joint_vel_rad_s", 1.50)),
            base_limit_rad=float(teleop_cfg.get("base_limit_rad", 2.6)),
            base_limit_damping_threshold=float(teleop_cfg.get("base_limit_damping_threshold", 0.8)),
        )

    robot.connect(use_control=True)
    if sense is not None:
        sense.connect()
    gripper.connect()
    if ext_cams is not None:
        ext_cams.connect()

    previewer = None
    if show_preview:
        def _frame_provider():
            frames = {}
            if ext_cams is not None:
                frames.update(ext_cams.get_latest_frames())
            if wrist_key is not None:
                wf = gripper.get_wrist_frame()
                if wf is not None:
                    frames[wrist_key] = wf
            return frames

        preview_cfg = cfg.get("preview", {}) or {}
        previewer = CameraPreviewer(
            frame_provider=_frame_provider,
            title="Pika cameras (q to close)",
            fps=30,
            target_height=int(preview_cfg.get("height", 720)),
        )
        previewer.start()

    if replay_only:
        print("\n[teleop_only] Devices ready for replay.")
    else:
        print("\n[teleop_only] Devices ready. Pull the Pika trigger to ENGAGE.")
        print("[teleop_only] Press Ctrl+C in this terminal to quit.\n")

    if teleop is not None:
        teleop.start()
    user_quit = False
    try:
        if replay_only:
            _replay_trajectory(
                robot=robot,
                gripper=gripper,
                npz_path=args.replay_path,
                use=args.replay_use,
                replay_hz=float(args.replay_hz),
                speed=float(args.replay_speed),
                lookahead_s=servo_lookahead_s,
                gain=servo_gain,
            )
            user_quit = True
            return

        recorded_path = ""
        if args.record_path:
            meta = {
                "config_path": str(Path(args.config).resolve()),
                "robot": cfg.get("robot", {}),
                "pika_sense": sense_cfg,
                "pika_gripper": cfg.get("pika_gripper", {}),
                "robotiq_gripper": cfg.get("robotiq_gripper", {}),
                "gripper_mapping": cfg.get("gripper_mapping", {}),
                "gripper_backend": gripper_backend,
                "teleoperation": teleop_cfg,
            }
            recorded_path = _record_trajectory(
                robot=robot,
                gripper=gripper,
                teleop=teleop,
                out_path=args.record_path,
                record_hz=float(args.record_hz),
                meta=meta,
            )
            if args.replay_after_record:
                teleop.stop()
                _replay_trajectory(
                    robot=robot,
                    gripper=gripper,
                    npz_path=recorded_path,
                    use=args.replay_use,
                    replay_hz=float(args.replay_hz),
                    speed=float(args.replay_speed),
                    lookahead_s=servo_lookahead_s,
                    gain=servo_gain,
                )
                user_quit = True
                return
            user_quit = True
            return

        while True:
            if teleop.aborted:
                print(f"\n[teleop_only] ABORTED — not user-initiated.")
                print(f"[teleop_only] Reason: {teleop.abort_reason}")
                print("[teleop_only] What to check:")
                print("              - UR pendant: Protective Stop / Local "
                      "mode / E-stop?")
                print("              - USB cables to Pika Sense / Gripper")
                print("              - Gripper power / Robotiq URCap program")
                print("              - `dmesg | tail` for kernel USB events")
                break
            if previewer is not None:
                state_lbl = "ENGAGED" if teleop.is_teleop_active else "released"
                previewer.set_status(f"teleop={state_lbl}  "
                                      f"scale={position_scale:.2f}  "
                                      f"max_delta={max_delta_m:.2f}m")
                if previewer.closed_by_user:
                    user_quit = True
                    print("[teleop_only] Preview window closed — exiting.")
                    break
            time.sleep(0.1)
    except KeyboardInterrupt:
        user_quit = True
        print("\n[teleop_only] Stopping (user Ctrl+C)...")
    finally:
        if previewer is not None:
            previewer.stop()
        if teleop is not None:
            teleop.stop()
        try:
            robot.stop()
        except Exception:
            pass
        robot.disconnect()
        if sense is not None:
            sense.disconnect()
        gripper.disconnect()
        if ext_cams is not None:
            ext_cams.disconnect()
        print("[teleop_only] Done.")


if __name__ == "__main__":
    main()
