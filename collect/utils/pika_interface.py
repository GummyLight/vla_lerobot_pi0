"""
Pika gripper and Pika Sense teleoperation interface.

IMPORTANT: This module wraps the Pika SDK. You must install the Pika Python
package provided by your Pika hardware vendor and adjust the import paths
and method calls in the two classes below to match your exact SDK version.

Typical SDK install:
    pip install pika-sdk          # or the wheel file provided by Pika
    # OR if using the serial-based driver:
    pip install pyserial

The classes expose a stable interface used by collect_pika.py regardless
of the underlying SDK version.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np


# ======================================================================
# Pika Gripper
# ======================================================================

class PikaGripper:
    """
    Controls the Pika dexterous gripper and reads its built-in wrist camera.

    Adapt the SDK calls (marked with # ADAPT) to your Pika SDK version.
    """

    OPEN = 0.0
    CLOSE = 1.0

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        """
        Args:
            port: Serial port (e.g. '/dev/ttyUSB0', 'COM3').
            baudrate: Typically 115200 for Pika devices.
        """
        self.port = port
        self.baudrate = baudrate
        self._device = None
        self._camera = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._cam_running = False

    def connect(self):
        # ADAPT: replace with your Pika SDK gripper init
        try:
            from pika_sdk import PikaGripperDevice  # adjust import
            self._device = PikaGripperDevice(port=self.port, baudrate=self.baudrate)
            self._device.connect()
        except ImportError:
            # Fallback: raw serial if SDK unavailable
            import serial
            self._device = serial.Serial(self.port, self.baudrate, timeout=1.0)
            print("[PikaGripper] Using raw serial fallback — some features may be limited.")

        self._start_camera()
        print(f"[PikaGripper] Connected @ {self.port}")

    def _start_camera(self):
        """Start background thread to capture the wrist camera."""
        # ADAPT: Replace with SDK camera init if your Pika gripper exposes
        # its camera via the SDK. Otherwise, open as a UVC device (webcam).
        try:
            import cv2
            self._cam_cap = cv2.VideoCapture(0)  # ADAPT: correct device index
            if not self._cam_cap.isOpened():
                print("[PikaGripper] Wrist camera not found — frames will be empty.")
                return
            self._cam_running = True
            t = threading.Thread(target=self._cam_loop, daemon=True)
            t.start()
        except Exception as e:
            print(f"[PikaGripper] Camera init failed: {e}")

    def _cam_loop(self):
        while self._cam_running:
            ret, frame = self._cam_cap.read()
            if ret:
                with self._lock:
                    self._latest_frame = frame

    def move(self, position: float, speed: float = 0.5):
        """
        position: 0.0 (fully open) … 1.0 (fully closed)
        speed:    0.0 … 1.0
        """
        position = float(np.clip(position, 0.0, 1.0))
        # ADAPT: replace with your SDK's gripper move command
        try:
            self._device.set_position(position, speed=speed)
        except Exception:
            # Raw serial fallback: send a simple ASCII command
            cmd = f"G{int(position * 255):03d}\n".encode()
            self._device.write(cmd)

    def get_position(self) -> float:
        """Return normalized gripper opening 0.0 (open) … 1.0 (closed)."""
        # ADAPT
        try:
            return float(self._device.get_position())
        except Exception:
            return 0.0

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Return latest BGR frame from the wrist camera, or None."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def disconnect(self):
        self._cam_running = False
        try:
            if hasattr(self, "_cam_cap"):
                self._cam_cap.release()
        except Exception:
            pass
        try:
            self._device.disconnect()
        except Exception:
            pass
        print("[PikaGripper] Disconnected.")


# ======================================================================
# Pika Sense (teleoperation controller)
# ======================================================================

class PikaSense:
    """
    Reads teleoperation commands from the Pika Sense wrist controller.

    The Sense device streams:
        - end-effector pose delta  (dx, dy, dz, drx, dry, drz)
        - gripper command          (0.0 open … 1.0 closed)
        - button states

    Adapt the SDK calls (marked with # ADAPT) to your Pika SDK version.
    """

    def __init__(self, port: str = "/dev/ttyUSB1", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._device = None
        self._latest: Dict = {
            "delta_pose": np.zeros(6, dtype=np.float32),
            "gripper": 0.0,
            "button_a": False,
            "button_b": False,
        }
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Calibration offset: removed from all pose readings
        self._pose_offset = np.zeros(6, dtype=np.float32)

    def connect(self):
        # ADAPT: replace with your Pika Sense SDK init
        try:
            from pika_sdk import PikaSenseDevice  # adjust import
            self._device = PikaSenseDevice(port=self.port, baudrate=self.baudrate)
            self._device.connect()
        except ImportError:
            import serial
            self._device = serial.Serial(self.port, self.baudrate, timeout=0.01)
            print("[PikaSense] Using raw serial fallback.")

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[PikaSense] Connected @ {self.port}")

    def _read_loop(self):
        """Background thread: continuously reads and parses Sense data."""
        while self._running:
            try:
                # ADAPT: replace with your SDK's data fetch call
                data = self._read_packet()
                if data:
                    with self._lock:
                        self._latest = data
            except Exception:
                time.sleep(0.001)

    def _read_packet(self) -> Optional[Dict]:
        """
        Parse one data packet from the Sense device.
        ADAPT this method to your actual protocol / SDK.

        Expected return dict:
            delta_pose: np.ndarray shape (6,)  [dx,dy,dz,drx,dry,drz]  in metres/radians
            gripper:    float 0.0–1.0
            button_a:   bool
            button_b:   bool
        """
        # ADAPT: If using the official SDK, call something like:
        #   packet = self._device.read()
        #   return {
        #       "delta_pose": np.array(packet.delta_pose),
        #       "gripper": packet.gripper,
        #       "button_a": packet.button_a,
        #       "button_b": packet.button_b,
        #   }
        #
        # Raw serial fallback stub (returns zeros — replace with real parsing):
        try:
            line = self._device.readline().decode().strip()
            if not line:
                return None
            parts = list(map(float, line.split(",")))
            if len(parts) < 7:
                return None
            return {
                "delta_pose": np.array(parts[:6], dtype=np.float32),
                "gripper": float(np.clip(parts[6], 0.0, 1.0)),
                "button_a": bool(parts[7]) if len(parts) > 7 else False,
                "button_b": bool(parts[8]) if len(parts) > 8 else False,
            }
        except Exception:
            return None

    def calibrate(self):
        """
        Hold the Sense at the desired neutral position and call this to zero it.
        Subsequent get_delta_pose() calls return motion relative to this pose.
        """
        print("[PikaSense] Calibrating — hold Sense still...")
        time.sleep(0.5)
        samples = []
        for _ in range(50):
            with self._lock:
                samples.append(self._latest["delta_pose"].copy())
            time.sleep(0.02)
        self._pose_offset = np.mean(samples, axis=0)
        print(f"[PikaSense] Calibrated. Offset: {self._pose_offset}")

    def get_latest(self) -> Dict:
        """Return the most recent Sense reading (thread-safe)."""
        with self._lock:
            d = dict(self._latest)
            d["delta_pose"] = d["delta_pose"] - self._pose_offset
        return d

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._device.disconnect()
        except Exception:
            pass
        print("[PikaSense] Disconnected.")
