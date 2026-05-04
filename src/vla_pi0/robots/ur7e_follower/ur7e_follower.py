"""UR7e + Robotiq 2F-58 + 2× RealSense D435i, exposed as a lerobot ``Robot``.

Implements the same `connect / disconnect / get_observation / send_action /
observation_features / action_features / is_connected / is_calibrated`
contract as upstream `lerobot.robots.robot.Robot`, so this rig can be
driven by `lerobot-rollout`-style scripts without any rig-specific
plumbing in the script itself.

Internally it delegates to the existing collectors in `collect/utils/`:
  - `UR7eInterface`  — RTDE state + control
  - `RobotiqGripper` — TCP socket to the Robotiq URCap server
  - per-camera RealSense pipes (one per ``CameraSpec``)
so behaviour is identical to the legacy `deploy/run_pi0_robot.py` path
while we migrate the rest of the stack.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

# Allow `from collect.utils...` to resolve even when this module is imported
# from outside the project root (e.g. `python -m vla_pi0.scripts.rollout`).
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collect.utils.robot_interface import UR7eInterface  # noqa: E402
from collect.utils.robotiq_interface import RobotiqGripper  # noqa: E402

from .config_ur7e_follower import UR7eFollowerConfig

# lerobot.Robot base — fall back to a structural stub if lerobot isn't on
# the path (lets unit tests of this file work in isolation).
try:
    from lerobot.robots.robot import Robot  # type: ignore
except Exception:  # pragma: no cover — tooling fallback
    class Robot:  # type: ignore[no-redef]
        config_class = None
        name = "robot"

        def __init__(self, config):
            self.config = config


class UR7eFollower(Robot):
    """6-DoF UR7e + Robotiq 2F jaw + 2× D435i cameras as a lerobot Robot."""

    config_class = UR7eFollowerConfig
    name = "ur7e_follower"

    def __init__(self, config: UR7eFollowerConfig):
        # Robot.__init__ expects (config) on recent lerobot; safe to ignore
        # if our shim base class skips super().__init__.
        try:
            super().__init__(config)
        except TypeError:
            self.config = config
        self.config: UR7eFollowerConfig = config

        self._robot: UR7eInterface | None = None
        self._gripper: RobotiqGripper | None = None
        self._cam_pipes: dict[str, Any] = {}
        self._cam_executor: ThreadPoolExecutor | None = None
        self._connected = False
        self._prev_target_joints: np.ndarray | None = None
        self._prev_loop_dt: float = 1.0 / config.control_hz

    # ------------------------------------------------------------------ Properties
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        # UR + Robotiq self-calibrate at power-on / activation; nothing to do here.
        return True

    @property
    def observation_features(self) -> dict[str, Any]:
        """Match the keys the dataset was recorded with.

        Joint positions and the gripper position are flat float scalars, one
        key per joint, so lerobot's recorder serialises them into the
        expected 7-d `observation.state` vector. Cameras are HxWx3 uint8 RGB.
        """
        feat: dict[str, Any] = {f"{j}.pos": float for j in self.config.joint_names}
        feat["gripper.pos"] = float
        for c in self.config.cameras:
            feat[c.name] = (c.height, c.width, 3)
        return feat

    @property
    def action_features(self) -> dict[str, Any]:
        feat: dict[str, Any] = {f"{j}.pos": float for j in self.config.joint_names}
        feat["gripper.pos"] = float
        return feat

    # ------------------------------------------------------------------ Connection
    def connect(self, calibrate: bool = False) -> None:
        if self._connected:
            return

        self._robot = UR7eInterface(
            host=self.config.ip,
            frequency=self.config.rtde_frequency,
        )
        self._robot.connect(use_control=True)

        self._gripper = RobotiqGripper(
            host=self.config.ip, port=self.config.gripper_port
        )
        self._gripper.connect()

        self._open_cameras()
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if self._cam_executor is not None:
                self._cam_executor.shutdown(wait=False, cancel_futures=True)
        finally:
            for pipe in self._cam_pipes.values():
                try:
                    pipe.stop()
                except Exception:
                    pass
            self._cam_pipes = {}
            self._cam_executor = None

        try:
            if self._robot is not None:
                self._robot.disconnect()
        finally:
            self._robot = None

        try:
            if self._gripper is not None:
                self._gripper.disconnect()
        finally:
            self._gripper = None

        self._connected = False
        self._prev_target_joints = None

    def configure(self) -> None:
        """No-op — RealSense + RTDE need no extra runtime tuning beyond what
        connect() already does. Present for lerobot.Robot interface parity."""
        return None

    def calibrate(self) -> None:
        """No-op — see `is_calibrated`."""
        return None

    def __enter__(self) -> "UR7eFollower":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # ------------------------------------------------------------------ Cameras
    def _open_cameras(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pyrealsense2 not installed. `pip install pyrealsense2`."
            ) from e

        # Two threads so per-camera wait_for_frames() runs in parallel.
        self._cam_executor = ThreadPoolExecutor(
            max_workers=max(2, len(self.config.cameras)),
            thread_name_prefix="rs",
        )
        # Resolve missing serials by enumeration order, but warn loudly —
        # this is fragile across reboots.
        ctx_serials: list[str] | None = None
        for cam in self.config.cameras:
            serial = cam.serial
            if not serial:
                if ctx_serials is None:
                    ctx_serials = [
                        d.get_info(rs.camera_info.serial_number)
                        for d in rs.context().devices
                    ]
                    print(
                        f"⚠ no camera serials given; auto-picking from "
                        f"enumeration order: {ctx_serials}"
                    )
                if not ctx_serials:
                    raise RuntimeError("no RealSense devices enumerated")
                serial = ctx_serials.pop(0)

            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color, cam.width, cam.height, rs.format.rgb8, cam.fps
            )
            pipe = rs.pipeline()
            pipe.start(cfg)
            self._cam_pipes[cam.name] = pipe

        # Drop a few warmup frames per pipe so AE/AWB stabilise.
        for _ in range(5):
            for pipe in self._cam_pipes.values():
                pipe.wait_for_frames()

    @staticmethod
    def _read_one(pipe) -> np.ndarray:
        f = pipe.wait_for_frames().get_color_frame()
        if not f:
            raise RuntimeError("RealSense returned empty color frame")
        return np.asanyarray(f.get_data())

    # ------------------------------------------------------------------ I/O
    def get_observation(self) -> dict[str, Any]:
        if not self._connected or self._robot is None:
            raise RuntimeError("UR7eFollower.get_observation() before connect()")

        state = self._robot.get_state()
        joints = state["joint_positions"]
        gripper_pos = self._gripper.get_position() if self._gripper else 0.0

        # Capture cameras concurrently so the slowest pipe sets the period.
        futures = {
            name: self._cam_executor.submit(self._read_one, pipe)
            for name, pipe in self._cam_pipes.items()
        }

        obs: dict[str, Any] = {}
        for i, jname in enumerate(self.config.joint_names):
            obs[f"{jname}.pos"] = float(joints[i])
        obs["gripper.pos"] = float(gripper_pos)
        for name, fut in futures.items():
            obs[name] = fut.result()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send a 7-d action (6 joint targets + gripper) and return the actually-
        commanded values (after safety clamping). Matches lerobot's contract:
        ``action_sent[k] == action[k]`` only when no clamp / clip kicked in."""
        if not self._connected or self._robot is None:
            raise RuntimeError("UR7eFollower.send_action() before connect()")

        target_joints = np.array(
            [float(action[f"{j}.pos"]) for j in self.config.joint_names],
            dtype=np.float32,
        )
        target_gripper = float(action["gripper.pos"])

        # Per-step joint delta safety cap.
        if self._prev_target_joints is not None:
            raw_delta = target_joints - self._prev_target_joints
            max_abs = float(np.abs(raw_delta).max())
            if max_abs > self.config.max_joint_delta_rad:
                if self.config.clamp_mode == "clip":
                    scale = self.config.max_joint_delta_rad / max_abs
                    target_joints = self._prev_target_joints + raw_delta * scale
                else:  # "refuse"
                    print(
                        f"⚠ joint delta {max_abs:.3f} rad > "
                        f"{self.config.max_joint_delta_rad}; holding previous target"
                    )
                    target_joints = self._prev_target_joints.copy()

        # servoJ time should match the actual loop period — see
        # collect/.../README and the legacy script's notes.
        sj = (
            self.config.servoj_velocity,
            self.config.servoj_acceleration,
            max(self._prev_loop_dt, 1.0 / self.config.control_hz),
            self.config.servoj_lookahead_time,
            self.config.servoj_gain,
        )
        self._robot._rtde_c.servoJ(target_joints.tolist(), *sj)

        if self._gripper is not None:
            jaw = 1.0 if target_gripper > self.config.gripper_threshold else 0.0
            self._gripper.write_position(jaw)

        self._prev_target_joints = target_joints
        # Caller is expected to call note_loop_dt() once per loop tick to
        # keep servoJ's reach time aligned with the actual loop period.
        action_sent: dict[str, Any] = {
            f"{j}.pos": float(target_joints[i])
            for i, j in enumerate(self.config.joint_names)
        }
        action_sent["gripper.pos"] = float(target_gripper)
        return action_sent

    def note_loop_dt(self, dt: float) -> None:
        """Tell the robot the actual measured period of the last control tick.

        servoJ takes a `time` argument that should match the control period;
        if we always pass `1/control_hz` but the loop is slow, the arm jerks
        because it tries to reach the target faster than the next command
        arrives. The rollout script measures dt and feeds it back here.
        """
        if dt > 0:
            self._prev_loop_dt = dt

    def servo_stop(self) -> None:
        if self._robot is not None and self._robot._rtde_c is not None:
            try:
                self._robot._rtde_c.servoStop()
            except Exception as e:
                print(f"⚠ servoStop failed: {e}")
