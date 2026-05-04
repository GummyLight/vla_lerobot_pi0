"""
Robotiq 2F-58 gripper interface via URCap socket server.

The Robotiq URCap exposes a Modbus-over-socket server on the UR controller
at port 63352. This module communicates directly with that server.

Install: pip install pymodbus (optional, used as fallback)
Primary method: direct socket to the Robotiq URCap gripper socket server.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Optional


_ACT_CMD = b"SET ACT 1\n"
_GTO_CMD = b"SET GTO 1\n"


class RobotiqGripper:
    """
    Controls the Robotiq 2F-58 via the gripper URCap socket server.

    The URCap runs on the UR controller and listens on TCP port 63352.
    Connect to: host = UR robot IP, port = 63352.
    """

    OPEN = 0
    CLOSE = 255

    def __init__(self, host: str, port: int = 63352, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        self._activate()
        print(f"[Gripper] Robotiq 2F-58 connected @ {self.host}:{self.port}")

    def _send(self, cmd: bytes) -> str:
        with self._lock:
            self._sock.sendall(cmd)
            try:
                return self._sock.recv(1024).decode().strip()
            except socket.timeout:
                return ""

    def _activate(self):
        self._send(b"SET ACT 0\n")
        time.sleep(0.1)
        self._send(b"SET ACT 1\n")
        # Wait for activation to complete
        deadline = time.time() + 10.0
        while time.time() < deadline:
            resp = self._send(b"GET STA\n")
            if "STA 3" in resp:
                break
            time.sleep(0.1)
        self._send(b"SET GTO 1\n")

    def move(self, position: int, speed: int = 200, force: int = 150):
        """
        position: 0 (fully open) … 255 (fully closed)
        speed, force: 0–255
        """
        position = max(0, min(255, position))
        speed = max(0, min(255, speed))
        force = max(0, min(255, force))
        cmd = f"SET POS {position}\nSET SPE {speed}\nSET FOR {force}\nSET GTO 1\n".encode()
        self._send(cmd)

    def open(self, speed: int = 200, force: int = 150):
        self.move(self.OPEN, speed, force)

    def close(self, speed: int = 200, force: int = 200):
        self.move(self.CLOSE, speed, force)

    def get_position_raw(self) -> int:
        """Return raw position 0–255."""
        resp = self._send(b"GET POS\n")
        try:
            return int(resp.split()[-1])
        except (ValueError, IndexError):
            return 0

    def get_position(self) -> float:
        """Return normalized position 0.0 (open) … 1.0 (closed)."""
        return self.get_position_raw() / 255.0

    # ------------------------------------------------------------------
    # Aliases so the same class works for both the data collector
    # (which calls connect/get_position/disconnect) and the inference
    # loop in deploy/run_pi0_robot.py (read_position/write_position/close).
    # ------------------------------------------------------------------

    def read_position(self) -> float:
        return self.get_position()

    def write_position(self, value: float) -> None:
        """Send a normalised position in [0, 1]: 0=open, 1=closed."""
        raw = int(round(max(0.0, min(1.0, float(value))) * 255))
        self.move(raw)

    def disconnect(self):
        if self._sock:
            self._sock.close()
            self._sock = None
        print("[Gripper] Disconnected.")
