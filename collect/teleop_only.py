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
from utils.pika_interface import PikaGripper, PikaSense, detect_pika_ports  # noqa: E402
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
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sense_cfg = cfg.get("pika_sense", {})
    gripper_cfg = cfg.get("pika_gripper", {})
    teleop_cfg = cfg.get("teleoperation", {})

    sense_port_pref = sense_cfg.get("port") or ""
    gripper_port_pref = gripper_cfg.get("port") or ""
    if not sense_port_pref or not gripper_port_pref:
        sp, gp = detect_pika_ports(sense_port_pref or None,
                                   gripper_port_pref or None)
        sense_port_pref = sense_port_pref or sp
        gripper_port_pref = gripper_port_pref or gp
        print(f"[teleop_only] Auto-detected ports — sense={sense_port_pref}, "
              f"gripper={gripper_port_pref}")

    robot = UR7eInterface(host=cfg["robot"]["host"],
                          frequency=cfg["robot"].get("frequency", 500.0))
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
    wrist_key = wrist_cam["name"] if (wrist_cam and show_preview) else None

    gripper = PikaGripper(
        port=gripper_port_pref,
        wrist_camera_kind=(wrist_cam.get("kind", "realsense")
                           if (wrist_cam and show_preview) else "none"),
        wrist_realsense_serial=(wrist_cam.get("serial")
                                if (wrist_cam and show_preview) else None),
        wrist_fisheye_index=(wrist_cam.get("device_index", 0)
                             if wrist_cam else 0),
        wrist_width=(wrist_cam.get("width", 640) if wrist_cam else 640),
        wrist_height=(wrist_cam.get("height", 480) if wrist_cam else 480),
        wrist_fps=(wrist_cam.get("fps", 30) if wrist_cam else 30),
        enable_motor_on_connect=gripper_cfg.get("enable_motor", True),
    )

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
        servo_lookahead_s=float(teleop_cfg.get("servo_lookahead_s", 0.2)),
        servo_gain=float(teleop_cfg.get("servo_gain", 100.0)),
        max_lin_vel_m_s=float(teleop_cfg.get("max_lin_vel_m_s", 0.30)),
        max_ang_vel_rad_s=float(teleop_cfg.get("max_ang_vel_rad_s", 1.50)),
        max_joint_vel_rad_s=float(teleop_cfg.get("max_joint_vel_rad_s", 1.50)),
        base_limit_rad=float(teleop_cfg.get("base_limit_rad", 2.6)),
        base_limit_damping_threshold=float(teleop_cfg.get("base_limit_damping_threshold", 0.8)),
    )

    robot.connect(use_control=True)
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

    print("\n[teleop_only] Devices ready. Pull the Pika trigger to ENGAGE.")
    print("[teleop_only] Press Ctrl+C in this terminal to quit.\n")

    teleop.start()
    user_quit = False
    try:
        while True:
            if teleop.aborted:
                print(f"\n[teleop_only] ABORTED — not user-initiated.")
                print(f"[teleop_only] Reason: {teleop.abort_reason}")
                print("[teleop_only] What to check:")
                print("              - UR pendant: Protective Stop / Local "
                      "mode / E-stop?")
                print("              - USB cables to Pika Sense / Gripper")
                print("              - Pika Gripper 24V power supply")
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
        teleop.stop()
        try:
            robot.stop()
        except Exception:
            pass
        robot.disconnect()
        sense.disconnect()
        gripper.disconnect()
        if ext_cams is not None:
            ext_cams.disconnect()
        print("[teleop_only] Done.")


if __name__ == "__main__":
    main()
