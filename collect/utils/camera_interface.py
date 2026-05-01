"""
Intel RealSense D435i camera interface.

Install: pip install pyrealsense2 opencv-python
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class D435iCamera:
    """Single D435i camera with buffered frame capture."""

    def __init__(
        self,
        name: str,
        serial: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
    ):
        self.name = name
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self._pipeline = None
        self._config = None

    def connect(self):
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            cfg.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
        self._pipeline.start(cfg)
        # Warm-up: discard first few frames
        for _ in range(10):
            self._pipeline.wait_for_frames(timeout_ms=5000)
        serial_str = f" (S/N {self.serial})" if self.serial else ""
        print(f"[Camera:{self.name}] Connected{serial_str}")

    def get_frames(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (color_bgr, depth_uint16_or_None)."""
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = None
        if self.enable_depth:
            depth = np.asanyarray(frames.get_depth_frame().get_data())
        return color, depth

    def disconnect(self):
        if self._pipeline:
            self._pipeline.stop()
        print(f"[Camera:{self.name}] Disconnected.")


class MultiCamera:
    """
    Manages multiple D435i cameras with a background capture thread
    so that get_latest_frames() never blocks longer than one frame period.
    """

    def __init__(self, cam_configs: List[dict]):
        """
        cam_configs: list of dicts with keys:
            name, serial (optional), width, height, fps, enable_depth (optional)
        """
        self._cameras: List[D435iCamera] = [
            D435iCamera(
                name=c["name"],
                serial=c.get("serial", ""),
                width=c.get("width", 640),
                height=c.get("height", 480),
                fps=c.get("fps", 30),
                enable_depth=c.get("enable_depth", False),
            )
            for c in cam_configs
        ]
        self._latest: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def connect(self):
        for cam in self._cameras:
            cam.connect()
        # Seed with one frame so get_latest_frames() works immediately
        for cam in self._cameras:
            color, _ = cam.get_frames()
            with self._lock:
                self._latest[cam.name] = color

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            for cam in self._cameras:
                try:
                    color, _ = cam.get_frames()
                    with self._lock:
                        self._latest[cam.name] = color
                except Exception as e:
                    print(f"[Camera:{cam.name}] Frame error: {e}")

    def get_latest_frames(self) -> Dict[str, np.ndarray]:
        """Return the most recently captured BGR frame from each camera."""
        with self._lock:
            return {k: v.copy() for k, v in self._latest.items()}

    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        for cam in self._cameras:
            cam.disconnect()


class UVCCamera:
    """
    Generic UVC camera (e.g. Pika gripper built-in camera accessed as webcam).
    Falls back gracefully if the device index is wrong.
    """

    def __init__(self, name: str, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.name = name
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def connect(self):
        self._cap = cv2.VideoCapture(self.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"[Camera:{self.name}] Cannot open device index {self.device_index}")
        print(f"[Camera:{self.name}] UVC device {self.device_index} opened.")

    def get_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"[Camera:{self.name}] Frame read failed.")
        return frame

    def disconnect(self):
        if self._cap:
            self._cap.release()
