"""Simple bridge: read PikaSense encoder and drive PikaGripper motor.

Usage:
    python collect/sense_gripper_bridge.py [--sense-port /dev/ttyUSB0] [--gripper-port /dev/ttyUSB1]
    python collect/sense_gripper_bridge.py --toggle

Features to reduce latency:
- configurable loop `--rate` (Hz), default 50Hz
- configurable EMA `--alpha` (0..1), higher -> more immediate (less smoothing)
- `--quiet` to minimize console I/O overhead
- configurable toggle debounce `--debounce` (s)

By default the script maps `sense.get_encoder_rad()` -> `gripper.set_motor_angle(rad)`.
This module is intentionally independent of any UR robot code.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from utils.pika_interface import PikaGripper, PikaSense, detect_pika_ports


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SenseGripperBridge:
    def __init__(self, sense_port: str = "", gripper_port: str = "", toggle_mode: bool = False,
                 rate_hz: float = 50.0, alpha: float = 0.8, quiet: bool = True, debounce: float = 0.15,
                 enc_min: float = 0.0, enc_max: float = 0.8, angle_min: float = 0.0, angle_max: float = 0.8,
                 invert: bool = False, debug: bool = False):
        self.sense = PikaSense(port=sense_port)
        self.gripper = PikaGripper(port=gripper_port)
        self.toggle_mode = toggle_mode
        self._running = False
        # EMA state for smooth motor commands (helps with encoder jitter)
        self._last_cmd = 0.0
        # alpha closer to 1.0 -> more immediate (less smoothing)
        self._alpha = float(alpha)
        self._is_closed = False
        # loop rate and print control
        self._rate_hz = max(1.0, float(rate_hz))
        self._period_s = 1.0 / self._rate_hz
        self._quiet = bool(quiet)
        self._debounce = float(debounce)
        # mapping parameters: map encoder [enc_min, enc_max] -> motor angle [angle_min, angle_max]
        self.enc_min = float(enc_min)
        self.enc_max = float(enc_max)
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.invert = bool(invert)
        self.debug = bool(debug)
        self._frame_count = 0
        self._last_enc = None
        self._send_error_count = 0

    def connect(self) -> None:
        sense_port, gripper_port = detect_pika_ports(self.sense.port, self.gripper.port)
        if not self.sense.port:
            self.sense.port = sense_port
        if not self.gripper.port:
            self.gripper.port = gripper_port

        print(f"Connecting Sense @ {self.sense.port} and Gripper @ {self.gripper.port}...")
        self.sense.connect()
        self.gripper.connect()
        # Drain a first pose if it exists (non-blocking)
        time.sleep(0.05)
        print("Connected.")

    def run(self) -> None:
        self._running = True

        def _sigint(_signum, _frame):
            print("Interrupted, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, _sigint)
        print("Starting bridge loop (Ctrl-C to quit)")
        try:
            next_print = time.time() + 1.0
            while self._running:
                loop_start = time.time()
                if not self.sense.is_alive() or not self.gripper.is_alive():
                    if not self._quiet:
                        print("Device disconnected. Exiting loop.")
                    break

                if self.toggle_mode:
                    # Toggle on rising edge of trigger state
                    trigger = self.sense.get_command_state()
                    if trigger == 1 and not self._is_closed:
                        # Close
                        self.gripper.set_motor_angle(0.0)  # 0 rad == fully open by SDK convention
                        self._is_closed = True
                        if not self._quiet:
                            print("Trigger pressed: closing gripper")
                    elif trigger == 1 and self._is_closed:
                        # Open to a stored open angle (e.g., 0.4 rad)
                        open_angle = 0.4
                        self.gripper.set_motor_angle(open_angle)
                        self._is_closed = False
                        if not self._quiet:
                            print("Trigger pressed: opening gripper")
                    # Debounce/avoid repeating on same frame
                    time.sleep(self._debounce)
                    continue

                # Default mode: read encoder and map to motor angle
                enc = self.sense.get_encoder_rad()
                enc = float(enc)

                # Check for NaN/inf
                if not np.isfinite(enc):
                    if self.debug:
                        print(f"[Frame {self._frame_count}] encoder is {enc} (invalid)")
                    enc = self._last_enc if self._last_enc is not None else 0.0

                # Linear map from encoder range to motor angle range
                # clamp encoder to specified range
                enc_clamped = clamp(enc, self.enc_min, self.enc_max)
                denom = (self.enc_max - self.enc_min) if (self.enc_max - self.enc_min) != 0 else 1.0
                t = (enc_clamped - self.enc_min) / denom
                if self.invert:
                    t = 1.0 - t
                target_angle = self.angle_min + t * (self.angle_max - self.angle_min)

                # EMA smoothing on target angle
                cmd = self._alpha * target_angle + (1.0 - self._alpha) * self._last_cmd
                self._last_cmd = cmd

                # Send to gripper
                try:
                    self.gripper.set_motor_angle(cmd)
                    if self.debug:
                        print(f"[Frame {self._frame_count}] enc={enc:.4f} -> target={target_angle:.4f} -> cmd={cmd:.4f} ✓")
                except Exception as e:
                    self._send_error_count += 1
                    if self.debug or self._send_error_count % 10 == 0:
                        print(f"[Frame {self._frame_count}] ✗ set_motor_angle({cmd:.4f}) failed: {e}")

                self._last_enc = enc
                self._frame_count += 1

                # Conditionally print a single-line status once per second
                if not self._quiet and time.time() >= next_print:
                    print(f"encoder={enc:.3f} rad -> cmd={cmd:.3f} rad")
                    next_print = time.time() + 1.0

                # Maintain loop rate
                elapsed = time.time() - loop_start
                to_sleep = self._period_s - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
        finally:
            try:
                self.sense.disconnect()
            except Exception:
                pass
            try:
                self.gripper.disconnect()
            except Exception:
                pass
            print("Bridge stopped.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sense-port", default="", help="Serial port for Sense (e.g. /dev/ttyUSB0)")
    p.add_argument("--gripper-port", default="", help="Serial port for Gripper (e.g. /dev/ttyUSB1)")
    p.add_argument("--toggle", action="store_true", help="Use trigger toggle mode instead of encoder mapping")
    p.add_argument("--rate", type=float, default=50.0, help="Loop rate in Hz (default: 50)")
    p.add_argument("--alpha", type=float, default=0.8, help="EMA alpha (0..1) for smoothing; higher -> more immediate")
    p.add_argument("--quiet", action="store_true", help="Minimize console output (recommended)")
    p.add_argument("--debounce", type=float, default=0.15, help="Toggle debounce seconds (default: 0.15)")
    p.add_argument("--enc-min", type=float, default=0.0, help="Encoder min value (maps to motor min)")
    p.add_argument("--enc-max", type=float, default=0.8, help="Encoder max value (maps to motor max)")
    p.add_argument("--angle-min", type=float, default=0.0, help="Motor angle for encoder min (rad)")
    p.add_argument("--angle-max", type=float, default=0.8, help="Motor angle for encoder max (rad)")
    p.add_argument("--invert", action="store_true", help="Invert mapping (encoder max -> motor min)")
    p.add_argument("--debug", action="store_true", help="Enable detailed frame-by-frame logging")
    args = p.parse_args(argv)

    bridge = SenseGripperBridge(
        sense_port=args.sense_port,
        gripper_port=args.gripper_port,
        toggle_mode=args.toggle,
        rate_hz=args.rate,
        alpha=args.alpha,
        quiet=args.quiet,
        debounce=args.debounce,
        enc_min=args.enc_min,
        enc_max=args.enc_max,
        angle_min=args.angle_min,
        angle_max=args.angle_max,
        invert=args.invert,
        debug=args.debug,
    )
    bridge.connect()
    bridge.run()


if __name__ == "__main__":
    main()
