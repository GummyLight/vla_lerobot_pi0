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
