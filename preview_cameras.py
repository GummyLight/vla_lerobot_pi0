"""
Quick D435i preview tool — verify camera framing before running collect_*.py.

Usage:
    python preview_cameras.py                # preview the first camera found
    python preview_cameras.py --serial <D435i_serial>
    python preview_cameras.py --all          # all cameras side by side
    python preview_cameras.py --all --fps 15 # halve the USB bandwidth

Press 'q' in the preview window to quit.
"""

import argparse
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def list_devices():
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]


def describe_devices():
    """Print serial / firmware / USB-type for every detected D435i."""
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
        print(f"  - {info['name']}  S/N={info['serial']}")
        print(f"      firmware     : {info['firmware']}  (recommended {info['recommended_firmware']})")
        print(f"      usb_type     : {info['usb_type']}    <-- must be 3.x for color stream")
        if str(info['usb_type']).startswith("2"):
            print("      !! WARNING: link negotiated as USB 2 — color stream will likely "
                  "time out. Try a different cable / port or rear-panel USB 3 port.")
    return serials


def open_pipeline(serial: str, width: int, height: int, fps: int) -> rs.pipeline:
    p = rs.pipeline()
    c = rs.config()
    c.enable_device(serial)
    c.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    p.start(c)
    return p


def warmup(pipe: rs.pipeline, serial: str, n: int = 15):
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


def preview(serials, width, height, fps):
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

        print("[preview] Streaming. Press 'q' to quit.", flush=True)
        while True:
            imgs = []
            for s, p in pipes:
                f = p.wait_for_frames(timeout_ms=2000).get_color_frame()
                img = np.asanyarray(f.get_data())
                cv2.putText(img, s, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                imgs.append(img)
            cv2.imshow("D435i preview  (q=quit)", np.hstack(imgs))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
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
    args = ap.parse_args()

    found = describe_devices()
    if not found:
        print("[preview] No RealSense devices found. Plug in a D435i and retry.")
        sys.exit(1)

    if args.serial:
        if args.serial not in found:
            print(f"[preview] Serial {args.serial} not found. Available: {found}")
            sys.exit(1)
        targets = [args.serial]
    elif args.all:
        targets = found
    else:
        targets = [found[0]]

    preview(targets, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
