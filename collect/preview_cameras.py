"""
Quick D435i preview tool — verify camera framing before running collect_*.py.

Usage:
    python preview_cameras.py                # preview the first camera found
    python preview_cameras.py --serial 405622074939
    python preview_cameras.py --all          # all cameras side by side, labelled from config
    python preview_cameras.py --all --fps 15 # halve the USB bandwidth

Press 'q' in the preview window to quit. Press 's' to save a snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np

rs = None

try:
    import yaml
except Exception:
    yaml = None


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def require_rs():
    global rs
    if rs is not None:
        return rs
    try:
        import pyrealsense2 as _rs
    except ModuleNotFoundError:
        print("[preview] pyrealsense2 is not installed in this Python environment.")
        print("          Activate the collection environment, e.g. `conda activate vla-pi0`,")
        print("          then rerun the preview command.")
        raise SystemExit(2)
    rs = _rs
    return rs


def list_devices():
    rs = require_rs()
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]


def default_config_path() -> Path | None:
    """Find the camera config whether the script is run from repo root or collect/."""
    candidates = [
        Path.cwd() / "configs/pika_config.yaml",
        Path.cwd() / "collect/configs/pika_config.yaml",
        HERE / "configs/pika_config.yaml",
        REPO_ROOT / "collect/configs/pika_config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_camera_roles(config_path: str | None) -> dict[str, dict]:
    """Return serial -> camera config for role labels such as cam_global/cam_wrist."""
    if not config_path:
        return {}
    if yaml is None:
        print("[preview] PyYAML not available; camera role labels disabled.")
        return {}

    path = Path(config_path).expanduser()
    if not path.exists():
        print(f"[preview] Config not found: {path}; camera role labels disabled.")
        return {}

    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    roles = {}
    for cam in cfg.get("cameras", []) or []:
        serial = str(cam.get("serial") or "").strip()
        if not serial:
            continue
        roles[serial] = {
            "name": cam.get("name", serial),
            "type": cam.get("type", ""),
            "width": cam.get("width"),
            "height": cam.get("height"),
            "fps": cam.get("fps"),
        }
    return roles


def order_serials(serials: list[str], roles: dict[str, dict]) -> list[str]:
    """Show configured cameras first in YAML order, then any extra devices."""
    configured = [s for s in roles if s in serials]
    extras = [s for s in serials if s not in configured]
    return configured + extras


def label_for(serial: str, roles: dict[str, dict]) -> str:
    role = roles.get(serial)
    if not role:
        return f"unconfigured  S/N={serial}"
    return f"{role['name']}  S/N={serial}"


def describe_devices(roles=None):
    """Print serial / firmware / USB-type for every detected D435i."""
    rs = require_rs()
    roles = roles or {}
    ctx = rs.context()
    devs = list(ctx.devices)
    if not devs:
        return []
    print(f"[preview] Found {len(devs)} RealSense device(s):")
    serials = []
    for d in devs:
        s = d.get_info(rs.camera_info.serial_number)
        serials.append(s)
        info = {"serial": s}
        for key, attr in [
            ("name", rs.camera_info.name),
            ("firmware", rs.camera_info.firmware_version),
            ("recommended_firmware", rs.camera_info.recommended_firmware_version),
            ("usb_type", rs.camera_info.usb_type_descriptor),
            ("product_line", rs.camera_info.product_line),
        ]:
            try:
                info[key] = d.get_info(attr)
            except Exception:
                info[key] = "?"
        role = roles.get(s, {})
        role_txt = f"  role={role.get('name')}" if role else "  role=<not in config>"
        print(f"  - {info['name']}  S/N={info['serial']}{role_txt}")
        print(f"      firmware     : {info['firmware']}  (recommended {info['recommended_firmware']})")
        print(f"      usb_type     : {info['usb_type']}    <-- must be 3.x for color stream")
        if str(info['usb_type']).startswith("2"):
            print("      !! WARNING: link negotiated as USB 2 — color stream will likely "
                  "time out. Try a different cable / port or rear-panel USB 3 port.")
    return serials


def open_pipeline(serial: str, width: int, height: int, fps: int):
    rs = require_rs()
    p = rs.pipeline()
    c = rs.config()
    c.enable_device(serial)
    c.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    p.start(c)
    return p


def warmup(pipe, serial: str, n: int = 15):
    for _ in range(n):
        try:
            pipe.wait_for_frames(timeout_ms=10000)
        except RuntimeError as e:
            print()
            print(f"[preview] !! Camera {serial} is enumerated but produced no frames "
                  f"({e}).")
            print("[preview] This almost always means one of:")
            print("           1) USB link degraded to USB 2  -> swap cable / port")
            print("              (use a SHORT USB-3 cable straight to a rear motherboard")
            print("              blue port; avoid hubs and front-panel ports).")
            print("           2) Camera firmware too old      -> update via Intel")
            print("              RealSense Viewer (it will offer Update Firmware).")
            print("           3) Insufficient USB power       -> avoid unpowered hubs;")
            print("              try a different rear USB port on a different controller.")
            print("[preview] After a failed run the camera may DROP off the bus until you")
            print("          unplug and replug the USB cable.")
            raise SystemExit(2)


def draw_label(img, text, subtitle=""):
    """Readable overlay for camera role + serial."""
    pad = 8
    y0 = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    lines = [text] + ([subtitle] if subtitle else [])
    line_h = 26
    box_h = pad * 2 + line_h * len(lines)
    cv2.rectangle(img, (0, 0), (img.shape[1], box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.putText(img, line, (10, y0 + pad + line_h * i + 12),
                    font, scale, color, thickness, cv2.LINE_AA)


def stack_images(imgs, target_height: int | None):
    if not imgs:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    resized = []
    for img in imgs:
        if target_height:
            h, w = img.shape[:2]
            new_w = max(1, int(round(w * target_height / h)))
            img = cv2.resize(img, (new_w, target_height))
        resized.append(img)
    return np.hstack(resized)


def preview(serials, width, height, fps, roles=None, target_height=None, snapshot_dir="snapshots"):
    roles = roles or {}
    print(f"[preview] Opening {len(serials)} camera(s) at {width}x{height}@{fps}...")
    pipes = []
    try:
        for i, s in enumerate(serials):
            print(f"  [{i+1}/{len(serials)}] starting {s} ...", flush=True)
            pipes.append((s, open_pipeline(s, width, height, fps)))
            time.sleep(0.5)

        print("[preview] Warming up (auto-exposure)...", flush=True)
        for s, p in pipes:
            warmup(p, s, n=15)
            print(f"  warmed up {s}", flush=True)

        print("[preview] Streaming. Press 'q' to quit, 's' to save snapshot.", flush=True)
        while True:
            imgs = []
            for s, p in pipes:
                f = p.wait_for_frames(timeout_ms=2000).get_color_frame()
                img = np.asanyarray(f.get_data())
                role = roles.get(s, {})
                subtitle = "configured" if role else "not in config"
                if role:
                    cfg_parts = [
                        str(v) for v in (role.get("type"), role.get("width"), role.get("height"), role.get("fps"))
                        if v not in (None, "")
                    ]
                    if cfg_parts:
                        subtitle = "config: " + " ".join(cfg_parts)
                draw_label(img, label_for(s, roles), subtitle)
                imgs.append(img)
            canvas = stack_images(imgs, target_height)
            cv2.imshow("RealSense preview  (q=quit, s=snapshot)", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                out_dir = Path(snapshot_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = out_dir / f"camera_preview_{ts}.png"
                cv2.imwrite(str(out), canvas)
                print(f"[preview] Snapshot saved: {out}")
    finally:
        for _, p in pipes:
            try:
                p.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None, help="Specific camera serial to preview")
    ap.add_argument("--all", action="store_true", help="Preview every camera found")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--config", default=None,
                    help="YAML config with cameras list. Defaults to configs/pika_config.yaml if found.")
    ap.add_argument("--no-config", action="store_true",
                    help="Do not label/order cameras from config.")
    ap.add_argument("--target-height", type=int, default=480,
                    help="Resize each preview tile to this height before stacking; 0 disables resizing.")
    ap.add_argument("--snapshot-dir", default="snapshots",
                    help="Directory for snapshots saved with 's'.")
    args = ap.parse_args()

    config_path = None if args.no_config else args.config
    if config_path is None and not args.no_config:
        p = default_config_path()
        config_path = str(p) if p else None
    roles = load_camera_roles(config_path)
    if config_path:
        print(f"[preview] Camera labels from: {config_path}")

    found = describe_devices(roles)
    if not found:
        print("[preview] No RealSense devices found. Plug in a D435i and retry.")
        sys.exit(1)

    if args.serial:
        if args.serial not in found:
            print(f"[preview] Serial {args.serial} not found. Available: {found}")
            sys.exit(1)
        targets = [args.serial]
    elif args.all:
        targets = order_serials(found, roles)
    else:
        targets = [order_serials(found, roles)[0]]

    configured_missing = [s for s in roles if s not in found]
    if configured_missing:
        print(f"[preview] !! Configured camera(s) not detected: {configured_missing}")
    print("[preview] Preview order:")
    for i, s in enumerate(targets, 1):
        print(f"  [{i}] {label_for(s, roles)}")

    target_height = args.target_height if args.target_height > 0 else None
    preview(targets, args.width, args.height, args.fps, roles,
            target_height=target_height, snapshot_dir=args.snapshot_dir)


if __name__ == "__main__":
    main()
