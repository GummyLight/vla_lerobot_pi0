"""Config dataclass for the UR7e + Robotiq 2F-58 follower.

Registered as `ur7e_follower` on `lerobot.robots.config.RobotConfig`, so it
can be selected from CLI via `--robot.type=ur7e_follower` exactly the same
way as upstream lerobot's `so100_follower`, `koch_follower`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# lerobot is a hard dep of this project (see environment.yml). If it is not
# importable we fall back to a minimal local stub so this file is at least
# parseable for tests / IDE — register_subclass becomes a no-op.
try:
    from lerobot.robots.config import RobotConfig  # type: ignore
except Exception:  # pragma: no cover — tooling fallback
    import abc
    from dataclasses import dataclass as _dc

    @_dc(kw_only=True)
    class RobotConfig(abc.ABC):  # type: ignore[no-redef]
        id: str | None = None
        calibration_dir: str | None = None

        @classmethod
        def register_subclass(cls, _name: str):
            def _decorator(klass):
                return klass
            return _decorator

        @property
        def type(self) -> str:
            return getattr(self, "_choice_name", "ur7e_follower")


@dataclass
class CameraSpec:
    """Lightweight RealSense spec — kept here (rather than reusing lerobot's
    `RealSenseCameraConfig`) so the rig can run even on a lerobot install
    that hasn't built the realsense optional extras."""

    name: str
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    use_depth: bool = False


@RobotConfig.register_subclass("ur7e_follower")
@dataclass(kw_only=True)
class UR7eFollowerConfig(RobotConfig):
    # --- robot connection ---
    ip: str = "192.168.1.100"
    rtde_frequency: float = 500.0
    gripper_port: int = 63352

    # --- servoJ realtime control parameters ---
    control_hz: float = 30.0
    servoj_velocity: float = 0.5
    servoj_acceleration: float = 0.5
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 300

    # --- safety ---
    max_joint_delta_rad: float = 0.10
    """Per-control-step joint delta cap (rad). Above this, send_action either
    clips the step (clamp_mode='clip') or refuses and resends the previous
    target (clamp_mode='refuse'). Default is conservative (~5.7 deg/step)."""

    clamp_mode: str = "refuse"  # "refuse" | "clip"
    gripper_threshold: float = 0.5
    """Continuous policy outputs in [0,1] are binarized at this threshold
    before being sent to the Robotiq jaw (open/close)."""

    # --- cameras ---
    cameras: list[CameraSpec] = field(default_factory=lambda: [
        CameraSpec(name="cam_global"),
        CameraSpec(name="cam_wrist"),
    ])

    # --- dataset feature naming ---
    joint_names: tuple[str, ...] = (
        "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    )

    def __post_init__(self) -> None:
        if self.clamp_mode not in ("refuse", "clip"):
            raise ValueError(
                f"clamp_mode must be 'refuse' or 'clip', got {self.clamp_mode!r}"
            )
        # Pydantic-free quick sanity on cameras: we expect exactly the two
        # views the dataset was recorded with.
        names = [c.name for c in self.cameras]
        if "cam_global" not in names or "cam_wrist" not in names:
            raise ValueError(
                f"UR7eFollowerConfig.cameras must include both 'cam_global' "
                f"and 'cam_wrist'; got {names}"
            )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a dict compatible with the existing run_pi0_robot.yaml schema.

        Useful while we migrate the legacy `deploy/run_pi0_robot.py` path —
        the new lerobot-style scripts use the dataclass directly.
        """
        return {
            "robot": {
                "ip": self.ip,
                "gripper_port": self.gripper_port,
                "servoj": {
                    "velocity": self.servoj_velocity,
                    "acceleration": self.servoj_acceleration,
                    "lookahead_time": self.servoj_lookahead_time,
                    "gain": self.servoj_gain,
                },
            },
            "cameras": [
                {
                    "name": c.name,
                    "serial": c.serial,
                    "width": c.width,
                    "height": c.height,
                    "fps": c.fps,
                }
                for c in self.cameras
            ],
            "control": {
                "hz": self.control_hz,
                "max_joint_delta_rad": self.max_joint_delta_rad,
                "clamp_mode": self.clamp_mode,
                "gripper_threshold": self.gripper_threshold,
            },
        }
