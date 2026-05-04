"""Pre-flight hardware check for run_pi0_robot.py.

Runs in seconds (no model load) and verifies, in order:
  1. UR controller TCP reachability + RTDE handshake.
  2. Robotiq URCap socket on UR (default port 63352).
  3. Both RealSense devices enumerable AND the configured serials are present.
  4. One color frame captured from each RealSense at the configured resolution.

Exit code 0 on full pass, 1 on any failure. Run this BEFORE any non-dry-run
launch of run_pi0_robot.py.

Usage:
    python deploy/preflight_check.py
    python deploy/preflight_check.py --config deploy/configs/run_pi0_robot.yaml
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "deploy/configs/run_pi0_robot.yaml"


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def check_robot_tcp(ip: str, timeout: float = 2.0) -> bool:
    print(f"[1/4] UR controller @ {ip}")
    # 30004 = RTDE port. Just opening it confirms the controller is up and
    # the network path is fine; full RTDE handshake happens in step 2.
    try:
        with socket.create_connection((ip, 30004), timeout=timeout):
            _ok(f"TCP {ip}:30004 (RTDE) reachable")
        return True
    except OSError as e:
        _fail(f"cannot reach {ip}:30004 — {e}")
        _fail("check: cable, IP, controller power, firewall, Remote-Control mode")
        return False


def check_rtde(ip: str) -> bool:
    print(f"[2/4] RTDE handshake @ {ip}")
    try:
        import rtde_receive
    except ImportError as e:
        _fail(f"ur_rtde not installed: {e}")
        return False
    try:
        recv = rtde_receive.RTDEReceiveInterface(ip)
        q = recv.getActualQ()
        _ok(f"RTDE handshake OK; current joints (rad) = {[round(v, 3) for v in q]}")
        del recv
        return True
    except RuntimeError as e:
        _fail(f"RTDE failed: {e}")
        _fail("check: pendant in Remote Control mode, no other RTDE client connected")
        return False


def check_robotiq(ip: str, port: int) -> bool:
    print(f"[3/4] Robotiq URCap socket @ {ip}:{port}")
    try:
        with socket.create_connection((ip, port), timeout=2.0) as s:
            s.sendall(b"GET POS\n")
            reply = s.recv(64).decode("ascii", errors="replace").strip()
        _ok(f"socket OK; reply to 'GET POS' = {reply!r}")
        if not reply.startswith("POS"):
            _fail("unexpected reply — URCap may not be running its program; "
                  "load and run a program containing the Robotiq toolbar")
            return False
        return True
    except OSError as e:
        _fail(f"cannot reach {ip}:{port} — {e}")
        _fail("check: Robotiq URCap installed AND a program containing the "
              "Robotiq toolbar is running on the pendant")
        return False


def check_realsense(cams_cfg: list[dict]) -> bool:
    print("[4/4] RealSense devices")
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        _fail(f"pyrealsense2 not installed: {e}")
        return False

    found = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name)
             for d in rs.context().devices}
    _ok(f"enumerated {len(found)} device(s): {found}")

    all_ok = True
    for c in cams_cfg:
        name = c.get("name", "?")
        serial = c.get("serial")
        w, h, fps = c.get("width", 640), c.get("height", 480), c.get("fps", 30)
        if not serial:
            _fail(f"{name}: no serial in config")
            all_ok = False
            continue
        if serial not in found:
            _fail(f"{name}: serial {serial} NOT plugged in")
            all_ok = False
            continue
        # Capture one frame at the configured resolution to confirm the stream works.
        try:
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
            pipe = rs.pipeline()
            pipe.start(cfg)
            try:
                f = pipe.wait_for_frames(timeout_ms=3000).get_color_frame()
                if not f:
                    _fail(f"{name} ({serial}): no color frame received")
                    all_ok = False
                else:
                    _ok(f"{name} ({serial}): captured {w}x{h}@{fps}fps RGB OK")
            finally:
                pipe.stop()
        except RuntimeError as e:
            _fail(f"{name} ({serial}): stream failed — {e}")
            all_ok = False
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--skip-robot", action="store_true",
                    help="Skip UR/Robotiq checks (cameras only).")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    print(f"using config: {args.config}\n")

    results: list[bool] = []

    if not args.skip_robot:
        ip = (cfg.get("robot") or {}).get("ip")
        gport = (cfg.get("robot") or {}).get("gripper_port", 63352)
        if not ip or ip == "192.168.1.100":
            _fail(f"robot.ip looks like the placeholder ({ip!r}); edit the config first")
            return 1
        results.append(check_robot_tcp(ip))
        if results[-1]:
            results.append(check_rtde(ip))
            results.append(check_robotiq(ip, gport))

    cams_cfg = cfg.get("cameras") or []
    results.append(check_realsense(cams_cfg))

    print()
    if all(results):
        print("PREFLIGHT PASSED — you can run deploy/run_pi0_robot.py now.")
        return 0
    print("PREFLIGHT FAILED — fix the items above before running on the real robot.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
