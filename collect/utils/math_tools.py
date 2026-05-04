"""
Pose math utilities used by the Pika teleoperation pipeline.

Adapted from pika_remote_ur/tools.py — the original pulls in
``tf.transformations`` (a ROS dependency) just to expose helpers that aren't
actually exercised by the teleop / collect path. This module keeps only the
methods that are used and reimplements them with NumPy/math, so it works in
any Python env without ROS installed.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class MathTools:
    """Stateless conversions between (x, y, z, roll, pitch, yaw), 4×4 matrices,
    rotation vectors and quaternions. Convention matches the upstream Pika
    teleop code (Z-Y-X intrinsic Tait–Bryan, radians)."""

    # ------------------------------------------------------------------
    # XYZ-RPY  ↔  4×4
    # ------------------------------------------------------------------

    def xyzrpy2Mat(self, x: float, y: float, z: float,
                   roll: float, pitch: float, yaw: float) -> np.ndarray:
        T = np.eye(4)
        A = np.cos(yaw);   B = np.sin(yaw)
        C = np.cos(pitch); D = np.sin(pitch)
        E = np.cos(roll);  F = np.sin(roll)
        DE = D * E
        DF = D * F
        T[0, 0] = A * C
        T[0, 1] = A * DF - B * E
        T[0, 2] = B * F + A * DE
        T[0, 3] = x
        T[1, 0] = B * C
        T[1, 1] = A * E + B * DF
        T[1, 2] = B * DE - A * F
        T[1, 3] = y
        T[2, 0] = -D
        T[2, 1] = C * F
        T[2, 2] = C * E
        T[2, 3] = z
        return T

    def mat2xyzrpy(self, matrix: np.ndarray) -> list:
        x = matrix[0, 3]
        y = matrix[1, 3]
        z = matrix[2, 3]
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.asin(-matrix[2, 0])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
        return [x, y, z, roll, pitch, yaw]

    # ------------------------------------------------------------------
    # RPY  ↔  rotation vector  (UR servoL uses rotvec)
    # ------------------------------------------------------------------

    def rpy_to_rotvec(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll),  np.cos(roll)]])
        R_y = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x

        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta = np.arccos(cos_theta)
        if abs(theta) < 1e-10:
            return np.zeros(3)
        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
        return axis * theta

    def rotvec_to_rpy(self, rotvec) -> Tuple[float, float, float]:
        rotvec = np.asarray(rotvec, dtype=float)
        theta = np.linalg.norm(rotvec)
        if abs(theta) < 1e-10:
            return (0.0, 0.0, 0.0)
        axis = rotvec / theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0
        return (float(roll), float(pitch), float(yaw))

    # ------------------------------------------------------------------
    # Quaternion → RPY  (tracker exposes [x, y, z, w])
    # ------------------------------------------------------------------

    def quaternion_to_rpy(self, x: float, y: float, z: float, w: float
                          ) -> Tuple[float, float, float]:
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (float(roll), float(pitch), float(yaw))

    # ------------------------------------------------------------------
    # Quaternion utilities (for proper rotational smoothing — RPY EMA
    # has discontinuities at ±π wrap and near gimbal lock; quat slerp
    # interpolates on the unit 3-sphere and is continuous everywhere).
    #
    # All quaternions are stored as [x, y, z, w] (matches pysurvive /
    # ROS conventions).
    # ------------------------------------------------------------------

    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / n

    def slerp(self, q0, q1, alpha: float) -> np.ndarray:
        """Spherical linear interpolation, alpha=0 → q0, alpha=1 → q1.
        Both inputs must be unit quaternions in [x, y, z, w]."""
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        # Pick the short path on S^3.
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        # Linear blend when nearly aligned (avoids div-by-zero).
        if dot > 0.9995:
            out = (1.0 - alpha) * q0 + alpha * q1
            return self.quat_normalize(out)
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        a = math.sin((1.0 - alpha) * theta) / sin_theta
        b = math.sin(alpha * theta) / sin_theta
        return a * q0 + b * q1

    def rpy_to_quat(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """ZYX intrinsic Tait–Bryan → quaternion [x, y, z, w]. Matches
        the convention used by xyzrpy2Mat / mat2xyzrpy."""
        cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
        # Z (yaw) ⊗ Y (pitch) ⊗ X (roll)
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        return self.quat_normalize(np.array([x, y, z, w], dtype=float))

    @staticmethod
    def quat_angle_diff(q0, q1) -> float:
        """Return the rotation-angle (radians) between two unit quats."""
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        dot = abs(float(np.dot(q0, q1)))
        dot = max(-1.0, min(1.0, dot))
        return 2.0 * math.acos(dot)

    @staticmethod
    def axis_angle_step(q_from, q_to, max_angle: float) -> np.ndarray:
        """Slerp from q_from toward q_to, but cap the step at ``max_angle``
        radians. Returns a quaternion that is at most ``max_angle`` away
        from ``q_from``."""
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        dot = float(np.dot(q_from, q_to))
        if dot < 0.0:
            q_to = -q_to
            dot = -dot
        dot = max(-1.0, min(1.0, dot))
        theta = 2.0 * math.acos(dot)
        if theta <= max_angle or theta < 1e-9:
            return q_to
        # Take the slerp at fraction max_angle/theta.
        t = max_angle / theta
        # Reuse slerp via a fresh instance (static method context).
        # Inline slerp to avoid re-creating MathTools().
        if dot > 0.9995:
            out = (1.0 - t) * q_from + t * q_to
        else:
            half = math.acos(dot)
            sin_half = math.sin(half)
            a = math.sin((1.0 - t) * half) / sin_half
            b = math.sin(t * half) / sin_half
            out = a * q_from + b * q_to
        n = float(np.linalg.norm(out))
        return out / n if n > 1e-12 else q_from
