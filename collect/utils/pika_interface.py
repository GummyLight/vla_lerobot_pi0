"""
Thin wrappers around the vendored Pika SDK (collect/pika_sdk/pika).

Two devices are exposed:

* ``PikaSense``   — tracker pose + encoder + trigger button. Reads from a
  Vive Tracker (T20 by default) via pysurvive.
* ``PikaGripper`` — motor enable / set_motor_angle / state read, plus access
  to the wrist camera (RealSense D405 by default; falls back to fisheye).

The serial port for sense and gripper can be auto-detected from
``/dev/ttyUSB*`` (configurable). Both devices stream JSON over the serial
link; the SDK takes care of parsing.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Make the vendored SDK importable as top-level ``pika`` package
# (collect/pika_sdk/pika/...) without requiring `pip install -e`.
# ----------------------------------------------------------------------
_SDK_ROOT = Path(__file__).resolve().parent.parent / "pika_sdk"
if _SDK_ROOT.exists() and str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))


logger = logging.getLogger("pika_interface")

# The vendored Pika SDK's read thread keeps spamming the same I/O error
# at 10 Hz once the USB device disappears, drowning the operator's
# terminal in red until they Ctrl+C. We watch for the dead-port state
# ourselves and exit cleanly via is_alive(), so silence the SDK's
# ERROR-level chatter on the serial transport. INFO/WARNING from other
# pika.* loggers (sense / gripper / vive_tracker) stay visible.
logging.getLogger("pika.serial_comm").setLevel(logging.CRITICAL)


# ======================================================================
# Port detection
# ======================================================================

def detect_pika_ports(prefer_sense: Optional[str] = None,
                      prefer_gripper: Optional[str] = None
                      ) -> Tuple[str, str]:
    """Return (sense_port, gripper_port).

    Resolution order:
        1. Explicit env vars (PIKA_SENSE_PORT, PIKA_GRIPPER_PORT) win.
        2. Explicit ``prefer_*`` argument from config.
        3. Lowest-indexed ``/dev/ttyUSB*`` becomes sense, next becomes gripper.
        4. Hard fallback to ttyUSB0 / ttyUSB1.
    """
    sense = os.environ.get("PIKA_SENSE_PORT") or prefer_sense
    gripper = os.environ.get("PIKA_GRIPPER_PORT") or prefer_gripper
    if sense and gripper:
        return sense, gripper

    candidates = sorted(glob.glob("/dev/ttyUSB*"))
    if not sense:
        sense = candidates[0] if candidates else "/dev/ttyUSB0"
    if not gripper:
        # Pick the next port that isn't `sense`
        rest = [c for c in candidates if c != sense]
        gripper = rest[0] if rest else "/dev/ttyUSB1"
    return sense, gripper


# ======================================================================
# Pika Sense — tracker pose + encoder + trigger
# ======================================================================

class PikaSense:
    """Wraps ``pika.sense.Sense``.

    Tracker pose is delivered through a background thread populated by
    ``connect()``. Call :py:meth:`get_tracker_pose` from the teleop loop;
    it returns the latest cached reading without blocking on Survive.
    """

    def __init__(
        self,
        port: str = "",
        tracker_device: str = "T20",
        tracker_config: Optional[str] = None,
        tracker_lh_config: Optional[str] = None,
    ):
        self.port = port
        self.tracker_device = tracker_device
        self._tracker_config = tracker_config
        self._tracker_lh_config = tracker_lh_config
        self._sense = None

        self._latest_pose: Optional[Tuple[list, list]] = None  # (pos[3], quat[4])
        self._latest_pose_lock = threading.Lock()
        self._tracker_thread: Optional[threading.Thread] = None
        self._tracker_running = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        from pika.sense import Sense  # vendored SDK
        if not self.port:
            self.port, _ = detect_pika_ports()

        self._sense = Sense(self.port)
        if not self._sense.connect():
            raise RuntimeError(
                f"[PikaSense] Failed to connect to {self.port}. "
                "Check that the device is plugged in and that you have rw "
                "permission on the tty (e.g. `sudo usermod -aG dialout $USER`)."
            )

        # Wire up the Vive tracker if configured. This pulls in pysurvive
        # lazily — if it's not installed, get_tracker_pose() returns None
        # and the teleop loop logs a warning.
        if self._tracker_config or self._tracker_lh_config:
            self._sense.set_vive_tracker_config(
                config_path=self._tracker_config,
                lh_config=self._tracker_lh_config,
            )

        # Drain initial tracker pose into our cache.
        self._tracker_running = True
        self._tracker_thread = threading.Thread(
            target=self._tracker_loop, daemon=True
        )
        self._tracker_thread.start()

        print(f"[PikaSense] Connected @ {self.port} (tracker={self.tracker_device})")
        return True

    def disconnect(self):
        self._tracker_running = False
        if self._tracker_thread and self._tracker_thread.is_alive():
            self._tracker_thread.join(timeout=1.0)
        if self._sense is not None:
            try:
                self._sense.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # tracker
    # ------------------------------------------------------------------

    def _tracker_loop(self):
        """Background thread: poll the SDK for tracker pose at ~50 Hz."""
        while self._tracker_running:
            try:
                pose = self._sense.get_pose(self.tracker_device)
            except Exception as e:
                logger.debug(f"[PikaSense] tracker poll error: {e}")
                pose = None
            if pose is not None:
                with self._latest_pose_lock:
                    self._latest_pose = (list(pose.position), list(pose.rotation))
            time.sleep(0.02)

    def get_tracker_pose(self) -> Optional[Tuple[list, list]]:
        """Return (position[3], quaternion[xyzw]) or None if not yet ready."""
        with self._latest_pose_lock:
            return self._latest_pose

    def wait_for_tracker(self, timeout: float = 10.0) -> bool:
        """Block until the first tracker pose arrives (or timeout)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.get_tracker_pose() is not None:
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    # encoder + trigger
    # ------------------------------------------------------------------

    def get_encoder_rad(self) -> float:
        """Return the Sense hand encoder in radians (commanded gripper angle)."""
        if self._sense is None:
            return 0.0
        try:
            return float(self._sense.get_encoder_data().get("rad", 0.0))
        except Exception:
            return 0.0

    def get_command_state(self) -> int:
        """Return the trigger button state (toggles between 0 and 1)."""
        if self._sense is None:
            return 0
        try:
            return int(self._sense.get_command_state())
        except Exception:
            return 0

    def is_alive(self) -> bool:
        """True iff the Sense USB serial port is still usable.

        ``[Errno 5] 输入/输出错误`` (EIO) means the kernel detached the
        device — the SDK's read thread catches and logs it but never
        flips ``is_connected`` to False, so an upstream watchdog is the
        only way to notice. This poll touches ``in_waiting`` which raises
        the same EIO when the port is dead.
        """
        if self._sense is None:
            return False
        sc = getattr(self._sense, "serial_comm", None)
        if sc is None or not getattr(sc, "is_connected", False):
            return False
        ser = getattr(sc, "serial", None)
        if ser is None:
            return False
        try:
            _ = ser.in_waiting
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # camera passthrough  (used when the wrist camera is on the Sense)
    # ------------------------------------------------------------------

    def set_realsense_serial(self, serial: str):
        if self._sense is not None:
            self._sense.set_realsense_serial_number(serial)

    def get_realsense_camera(self):
        return self._sense.get_realsense_camera() if self._sense else None


# ======================================================================
# Pika Gripper — motor + wrist camera
# ======================================================================

class PikaGripper:
    """Wraps ``pika.gripper.Gripper``.

    Use :py:meth:`set_motor_angle` to drive the gripper from the Sense
    encoder reading at ~50 Hz. Wrist camera frames are fetched lazily on
    first call to :py:meth:`get_wrist_frame`.
    """

    def __init__(
        self,
        port: str = "",
        wrist_camera_kind: str = "realsense",   # "realsense" | "fisheye" | "none"
        wrist_realsense_serial: Optional[str] = None,
        wrist_fisheye_index: int = 0,
        wrist_width: int = 640,
        wrist_height: int = 480,
        wrist_fps: int = 30,
        enable_motor_on_connect: bool = True,
    ):
        self.port = port
        self.wrist_camera_kind = wrist_camera_kind
        self.wrist_realsense_serial = wrist_realsense_serial
        self.wrist_fisheye_index = wrist_fisheye_index
        self.wrist_width = wrist_width
        self.wrist_height = wrist_height
        self.wrist_fps = wrist_fps
        self.enable_motor_on_connect = enable_motor_on_connect

        self._gripper = None
        self._wrist_camera = None
        self._wrist_camera_failed = False

    def connect(self) -> bool:
        from pika.gripper import Gripper  # vendored SDK
        if not self.port:
            _, self.port = detect_pika_ports()

        self._gripper = Gripper(self.port)
        if not self._gripper.connect():
            raise RuntimeError(
                f"[PikaGripper] Failed to connect to {self.port}."
            )

        if self.enable_motor_on_connect:
            if self._gripper.enable():
                print("[PikaGripper] Motor enabled.")
            else:
                print("[PikaGripper] !! Motor enable returned False — check power.")

        # Configure wrist camera params on the SDK side; actual open is lazy.
        if self.wrist_camera_kind == "realsense":
            if self.wrist_realsense_serial:
                self._gripper.set_realsense_serial_number(self.wrist_realsense_serial)
            self._gripper.set_camera_param(
                self.wrist_width, self.wrist_height, self.wrist_fps,
            )
        elif self.wrist_camera_kind == "fisheye":
            self._gripper.set_fisheye_camera_index(self.wrist_fisheye_index)
            self._gripper.set_camera_param(
                self.wrist_width, self.wrist_height, self.wrist_fps,
            )

        print(f"[PikaGripper] Connected @ {self.port} (wrist_cam={self.wrist_camera_kind})")
        return True

    def disconnect(self):
        if self._gripper is None:
            return
        try:
            self._gripper.disable()
        except Exception:
            pass
        try:
            self._gripper.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # motor
    # ------------------------------------------------------------------

    def set_motor_angle(self, rad: float):
        if self._gripper is None:
            return
        # Sense encoder occasionally drifts to small negative values near the
        # fully-open position; the SDK clamps internally but logs WARNING per
        # call. Clamp here so the SDK sees a clean value and stays quiet.
        rad = max(0.0, float(rad))
        try:
            self._gripper.set_motor_angle(rad)
        except Exception as e:
            logger.debug(f"[PikaGripper] set_motor_angle error: {e}")

    def get_motor_position(self) -> float:
        """Return the current motor position in radians (0 = fully open).

        Sanitizes inf / NaN / pathological values that occasionally surface
        when the SDK's serial parser drops a frame and reads past the end of
        a previous JSON packet — those flow into LeRobot's float32 cast and
        crash the ffmpeg writer at end-of-episode if not caught.
        """
        if self._gripper is None:
            return 0.0
        try:
            v = float(self._gripper.get_motor_position())
        except Exception:
            return 0.0
        if not np.isfinite(v) or abs(v) > 1e3:
            return 0.0
        return v

    def get_distance_mm(self) -> float:
        """Return current finger-tip opening distance in millimetres."""
        if self._gripper is None:
            return 0.0
        try:
            return float(self._gripper.get_gripper_distance())
        except Exception:
            return 0.0

    def is_alive(self) -> bool:
        """True iff the Gripper USB serial port is still usable.
        See ``PikaSense.is_alive`` for rationale.
        """
        if self._gripper is None:
            return False
        sc = getattr(self._gripper, "serial_comm", None)
        if sc is None or not getattr(sc, "is_connected", False):
            return False
        ser = getattr(sc, "serial", None)
        if ser is None:
            return False
        try:
            _ = ser.in_waiting
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # wrist camera  (RealSense D405 lives on the gripper PCB)
    # ------------------------------------------------------------------

    def _ensure_wrist_camera(self):
        if self._wrist_camera is not None or self._wrist_camera_failed:
            return self._wrist_camera
        if self._gripper is None or self.wrist_camera_kind == "none":
            return None
        try:
            if self.wrist_camera_kind == "realsense":
                self._wrist_camera = self._gripper.get_realsense_camera()
            elif self.wrist_camera_kind == "fisheye":
                self._wrist_camera = self._gripper.get_fisheye_camera()
            else:
                self._wrist_camera = None
        except Exception as e:
            print(f"[PikaGripper] wrist camera init failed: {e}")
            self._wrist_camera = None

        if self._wrist_camera is None:
            self._wrist_camera_failed = True
        return self._wrist_camera

    def get_wrist_frame(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame from the wrist camera, or None."""
        cam = self._ensure_wrist_camera()
        if cam is None:
            return None
        # Most pika camera classes expose either .get_color_frame() (RealSense)
        # or .get_frame() (fisheye). Try both, then fall back to whatever the
        # object supports — keeps this resilient against minor SDK churn.
        for attr in ("get_color_frame", "get_frame", "read"):
            fn = getattr(cam, attr, None)
            if callable(fn):
                try:
                    frame = fn()
                except Exception as e:
                    logger.debug(f"[PikaGripper] wrist {attr}() failed: {e}")
                    return None
                if frame is None:
                    return None
                # Some SDKs return (ret, frame) — unwrap.
                if isinstance(frame, tuple):
                    frame = frame[-1]
                return frame
        return None
