"""Gripper backends for PikaSense teleoperation.

The teleop loop only knows how to read the PikaSense encoder. These adapters
translate that encoder command to either the original Pika gripper motor angle
or a Robotiq normalized position command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.pika_interface import PikaGripper
from utils.robotiq_interface import RobotiqGripper


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class GripperMapping:
    pika_closed_rad: float = 0.0
    pika_open_rad: float = 1.7
    robotiq_closed_pos: float = 1.0
    robotiq_open_pos: float = 0.0
    deadband: float = 0.01

    @classmethod
    def from_config(cls, cfg: dict | None) -> "GripperMapping":
        cfg = cfg or {}
        return cls(
            pika_closed_rad=float(cfg.get("pika_closed_rad", 0.0)),
            pika_open_rad=float(cfg.get("pika_open_rad", 1.7)),
            robotiq_closed_pos=float(cfg.get("robotiq_closed_pos", 1.0)),
            robotiq_open_pos=float(cfg.get("robotiq_open_pos", 0.0)),
            deadband=float(cfg.get("deadband", 0.01)),
        )

    def pika_rad_to_robotiq(self, rad: float) -> float:
        denom = self.pika_open_rad - self.pika_closed_rad
        if abs(denom) < 1e-9:
            t = 0.0
        else:
            t = (float(rad) - self.pika_closed_rad) / denom
        t = clamp(t, 0.0, 1.0)
        pos = self.robotiq_closed_pos + t * (
            self.robotiq_open_pos - self.robotiq_closed_pos
        )
        return clamp(pos, 0.0, 1.0)


class PikaGripperAdapter:
    backend = "pika"

    def __init__(self, gripper: PikaGripper):
        self.gripper = gripper

    @property
    def port(self) -> str:
        return self.gripper.port

    @port.setter
    def port(self, value: str) -> None:
        self.gripper.port = value

    def connect(self) -> bool:
        return self.gripper.connect()

    def disconnect(self) -> None:
        self.gripper.disconnect()

    def is_alive(self) -> bool:
        return self.gripper.is_alive()

    def read_position(self) -> float:
        return self.gripper.get_motor_position()

    def command_from_pika_encoder(self, rad: float, dt: float) -> float:
        del dt
        cmd = max(0.0, float(rad))
        self.gripper.set_motor_angle(cmd)
        return cmd

    def set_replay_position(self, value: float) -> None:
        self.gripper.set_motor_angle(value)

    def get_wrist_frame(self) -> Optional[np.ndarray]:
        return self.gripper.get_wrist_frame()


class RobotiqGripperAdapter:
    backend = "robotiq"

    def __init__(
        self,
        host: str,
        port: int = 63352,
        mapping: Optional[GripperMapping] = None,
        force: int = 150,
        speed_min: int = 80,
        speed_max: int = 255,
        max_norm_speed_per_s: float = 2.0,
    ):
        self.gripper = RobotiqGripper(host=host, port=int(port))
        self.mapping = mapping or GripperMapping()
        self.force = int(force)
        self.speed_min = int(speed_min)
        self.speed_max = int(speed_max)
        self.max_norm_speed_per_s = max(1e-6, float(max_norm_speed_per_s))
        self._last_sent_pos: Optional[float] = None

    def connect(self) -> None:
        self.gripper.connect()

    def disconnect(self) -> None:
        self.gripper.disconnect()

    def is_alive(self) -> bool:
        return self.gripper.is_alive()

    def read_position(self) -> float:
        return self.gripper.read_position()

    def command_from_pika_encoder(self, rad: float, dt: float) -> float:
        pos = self.mapping.pika_rad_to_robotiq(rad)
        if (self._last_sent_pos is not None
                and abs(pos - self._last_sent_pos) < self.mapping.deadband):
            return self._last_sent_pos

        if self._last_sent_pos is None:
            speed = self.speed_max
        else:
            norm_speed = abs(pos - self._last_sent_pos) / max(float(dt), 1e-6)
            t = clamp(norm_speed / self.max_norm_speed_per_s, 0.0, 1.0)
            speed = int(round(self.speed_min + t * (self.speed_max - self.speed_min)))

        self.gripper.write_position(pos, speed=speed, force=self.force)
        self._last_sent_pos = pos
        return pos

    def set_replay_position(self, value: float) -> None:
        pos = clamp(float(value), 0.0, 1.0)
        self.gripper.write_position(pos, speed=self.speed_max, force=self.force)
        self._last_sent_pos = pos

    def get_wrist_frame(self) -> Optional[np.ndarray]:
        return None


def make_gripper_backend(
    backend: str,
    cfg: dict,
    *,
    wrist_cam: Optional[dict] = None,
    show_preview: bool = True,
):
    backend = (backend or "pika").lower()
    if backend == "robotiq":
        robotiq_cfg = cfg.get("robotiq_gripper") or cfg.get("gripper") or {}
        mapping = GripperMapping.from_config(cfg.get("gripper_mapping"))
        return RobotiqGripperAdapter(
            host=cfg["robot"]["host"],
            port=int(robotiq_cfg.get("port", 63352)),
            mapping=mapping,
            force=int(robotiq_cfg.get("force", 150)),
            speed_min=int(robotiq_cfg.get("speed_min", 80)),
            speed_max=int(robotiq_cfg.get("speed_max", 255)),
            max_norm_speed_per_s=float(
                robotiq_cfg.get("max_norm_speed_per_s", 2.0)
            ),
        )

    if backend != "pika":
        raise ValueError(f"Unsupported gripper backend: {backend!r}")

    gripper_cfg = cfg.get("pika_gripper", {})
    gripper = PikaGripper(
        port=gripper_cfg.get("port", "") or "",
        wrist_camera_kind=(wrist_cam.get("kind", "realsense")
                           if (wrist_cam and show_preview) else "none"),
        wrist_realsense_serial=(wrist_cam.get("serial")
                                if (wrist_cam and show_preview) else None),
        wrist_fisheye_index=(wrist_cam.get("device_index", 0)
                             if wrist_cam else 0),
        wrist_width=(wrist_cam.get("width", 640) if wrist_cam else 640),
        wrist_height=(wrist_cam.get("height", 480) if wrist_cam else 480),
        wrist_fps=(wrist_cam.get("fps", 30) if wrist_cam else 30),
        enable_motor_on_connect=gripper_cfg.get("enable_motor", True),
    )
    return PikaGripperAdapter(gripper)
