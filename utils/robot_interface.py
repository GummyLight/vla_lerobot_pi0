"""
UR7e robot interface via ur_rtde.

Install: pip install ur-rtde
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np


class UR7eInterface:
    """Thin wrapper around ur_rtde for state reading and motion control."""

    def __init__(self, host: str, frequency: float = 500.0):
        self.host = host
        self.frequency = frequency
        self._rtde_r = None
        self._rtde_c = None

    def connect(self):
        import rtde_control
        import rtde_receive

        self._rtde_r = rtde_receive.RTDEReceiveInterface(self.host, self.frequency)
        self._rtde_c = rtde_control.RTDEControlInterface(self.host)
        print(f"[Robot] Connected to UR7e @ {self.host}")

    def disconnect(self):
        try:
            if self._rtde_c:
                self._rtde_c.stopScript()
                self._rtde_c.disconnect()
            if self._rtde_r:
                self._rtde_r.disconnect()
        except Exception:
            pass
        print("[Robot] Disconnected.")

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": np.array(self._rtde_r.getActualQ(), dtype=np.float32),
            "joint_velocities": np.array(self._rtde_r.getActualQd(), dtype=np.float32),
            "tcp_pose": np.array(self._rtde_r.getActualTCPPose(), dtype=np.float32),
            "tcp_force": np.array(self._rtde_r.getActualTCPForce(), dtype=np.float32),
        }

    def is_steady(self, vel_threshold: float = 0.01) -> bool:
        """Return True when all joints are nearly stationary."""
        return float(np.max(np.abs(self._rtde_r.getActualQd()))) < vel_threshold

    # ------------------------------------------------------------------
    # Motion control
    # ------------------------------------------------------------------

    def send_urscript(self, script: str):
        """Send a raw URScript program string to the robot."""
        self._rtde_c.sendCustomScript(script)

    def send_urscript_file(self, path: str):
        with open(path) as f:
            script = f.read()
        self.send_urscript(script)

    def move_j(
        self,
        joints: List[float],
        speed: float = 0.5,
        acc: float = 0.5,
        asynchronous: bool = False,
    ):
        self._rtde_c.moveJ(joints, speed, acc, asynchronous)

    def move_l(
        self,
        pose: List[float],
        speed: float = 0.1,
        acc: float = 0.1,
        asynchronous: bool = False,
    ):
        self._rtde_c.moveL(pose, speed, acc, asynchronous)

    def servo_l(
        self,
        pose: List[float],
        speed: float = 0.5,
        acc: float = 0.5,
        dt: float = 0.002,
        lookahead: float = 0.1,
        gain: float = 300,
    ):
        """Streaming servo command for smooth teleoperation (call at >=100 Hz)."""
        self._rtde_c.servoL(pose, speed, acc, dt, lookahead, gain)

    def servo_stop(self):
        self._rtde_c.servoStop()

    def freedrive_mode(self, enable: bool):
        if enable:
            self._rtde_c.teachMode()
        else:
            self._rtde_c.endTeachMode()

    def stop(self, deceleration: float = 2.0):
        self._rtde_c.stopJ(deceleration)

    def is_program_running(self) -> bool:
        return self._rtde_r.isRobotMoving()
