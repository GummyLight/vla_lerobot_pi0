"""
Pika-Sense teleoperation + LeRobot v3.0 data collection.

Hardware path
-------------
    Vive Tracker (T20) ─┐
                         ├─► Pika Sense (USB serial, JSON)
    Hand encoder (rad) ─┘
                                      │  threaded loop @ 50 Hz
                                      ▼
                Pika Gripper ◄── set_motor_angle(rad)
                                      │
                                      ▼
                        UR7e ◄── servoL(target_pose_rotvec)

What we record (per frame, 30 fps default)
------------------------------------------
    observation.state  = [q0..q5, gripper_rad]            float32 [7]
    action             = [target_q0..target_q5, gripper_cmd_rad] float32 [7]
    observation.images.cam_global   ← external D435i   (MultiCamera)
    observation.images.cam_wrist    ← Pika wrist cam   (PikaGripper)

Output is LeRobot v3.0 — directly loadable by ``LeRobotDataset`` and ready
for pi0 fine-tuning without further conversion.

Usage
-----
    python collect_pika.py --config configs/pika_config.yaml \\
                           --dataset_name my_pika_dataset \\
                           --task "pour liquid"

    Press the Pika Sense trigger (single click) to enter teleop mode; the
    arm tracks the Sense pose deltas. Click again to release. Then in this
    terminal: [s] to start recording an episode, [Ctrl+C] to end it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

# Make the vendored pika SDK importable without `pip install`.
_HERE = Path(__file__).resolve().parent
_PIKA_SDK = _HERE / "pika_sdk"
if _PIKA_SDK.exists() and str(_PIKA_SDK) not in sys.path:
    sys.path.insert(0, str(_PIKA_SDK))

from utils.camera_interface import MultiCamera                           # noqa: E402
from utils.lerobot_writer import LeRobotWriter                            # noqa: E402
from utils.math_tools import MathTools                                    # noqa: E402
from utils.pika_interface import PikaGripper, PikaSense, detect_pika_ports  # noqa: E402
from utils.preview import CameraPreviewer                                  # noqa: E402
from utils.robot_interface import UR7eInterface                            # noqa: E402


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("collect_pika")


# ----------------------------------------------------------------------
# State / action space
# ----------------------------------------------------------------------
STATE_NAMES = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "gripper",
]
ACTION_NAMES = STATE_NAMES  # same dim — teleop drives commanded q + gripper


# ======================================================================
# Teleop controller (background thread)
# ======================================================================

class PikaTeleopController:
    """Reproduces the pika_remote_ur teleop logic against UR7e:

    * the Sense reports a 6-DoF pose from the Vive tracker (T20)
    * we apply the rigid pika→arm-end calibration to that pose
    * incremental motion since the last "trigger pressed" pose is added to
      the arm's initial TCP pose, then sent via servoL at SERVO_HZ
    * the Sense encoder rad is forwarded to the Pika gripper
    """

    def __init__(
        self,
        robot: UR7eInterface,
        sense: PikaSense,
        gripper: PikaGripper,
        pika_to_arm: list,
        position_scale: float = 1.0,
        max_delta_m: float = 1.0,
        servo_hz: int = 50,
        smoothing_alpha: float = 1.0,
        gripper_smoothing_alpha: float = 1.0,
        workspace_bounds: Optional[dict] = None,   # {"x":[lo,hi], ...}
        joint_limits: Optional[list] = None,       # 6 entries, each [lo,hi] or None
        max_tilt_from_down_rad: Optional[float] = None,
    ):
        self.robot = robot
        self.sense = sense
        self.gripper = gripper
        self.pika_to_arm = list(pika_to_arm)
        self.position_scale = float(position_scale)
        self.max_delta_m = float(max_delta_m)
        self.servo_hz = int(servo_hz)
        self.dt = 1.0 / self.servo_hz

        # EMA low-pass on the tracker pose (and gripper command) to damp
        # operator hand tremor before it ever reaches the arm. alpha=1 means
        # passthrough (no smoothing); alpha=0.2 means each new sample is
        # blended 20% with 80% of the previous filtered value (heavy
        # smoothing). Typical sweet spot for Vive→UR teleop: 0.25–0.4.
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.gripper_smoothing_alpha = float(np.clip(gripper_smoothing_alpha,
                                                     0.0, 1.0))
        self._smoothed_xyzrpy: Optional[np.ndarray] = None
        self._smoothed_gripper: Optional[float] = None

        self.tools = MathTools()

        # State of the teleoperation FSM
        self._initial_pose_rpy: Optional[list] = None    # arm "zero" snapshot (xyzrpy)
        self._base_pose: Optional[list] = None           # tracker reading at trigger-on
        self._tracker_xyzrpy: Optional[list] = None      # latest tracker reading mapped to arm frame
        self._teleop_active = False                       # mirrors trigger button toggle
        self._last_trigger = None
        self._last_gripper_cmd = 0.0

        # Failure detection — UR controller occasionally drops the RTDE
        # control script (protective stop, brake release loss, network
        # blip). Count consecutive failures and abort cleanly past a
        # threshold so the user sees a clear message instead of an
        # infinite spam of servoL errors.
        self._servo_fail_streak = 0
        self._servo_fail_limit = 30  # ~0.6s at 50 Hz
        self.controller_lost = False  # legacy alias kept for older callers
        self.aborted = False
        self.abort_reason: str = ""

        # ---------------- Safety filtering ----------------
        # Pre-flight check on every servoL target. Each sub-filter is
        # independently optional. When any returns "unsafe", the target is
        # rejected (no servoL sent) and the operator sees a throttled log.
        self.workspace_bounds = workspace_bounds or {}
        self.joint_limits = joint_limits  # None or 6-element list
        self.max_tilt_from_down_rad = max_tilt_from_down_rad
        self._reject_log_t = 0.0
        self._reject_count = 0

        # Worker thread
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # public API used by the recording loop
    # ------------------------------------------------------------------

    def start(self):
        # Snapshot the arm's current pose as the "zero" reference. Subsequent
        # teleop deltas are added to this pose.
        self._initial_pose_rpy = self._tcp_actual_xyzrpy()
        self._base_pose = list(self._initial_pose_rpy)
        # Wait for the tracker to deliver a first pose, otherwise we'd compute
        # nonsense increments on the first iteration. Lighthouse OOTX decode
        # + initial MPFIT solve commonly takes 15-25 s on a cold boot — give
        # it 30 s before complaining.
        if not self.sense.wait_for_tracker(timeout=30.0):
            print("[Teleop] !! Tracker pose not received in 30 s. Continuing "
                  "anyway — teleop will activate as soon as the tracker locks on.")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.robot.servo_stop()
        except Exception:
            pass

    def get_command_snapshot(self) -> Dict[str, float]:
        """Return the latest commanded gripper rad — read from the recording loop."""
        with self._lock:
            return {"gripper_cmd": float(self._last_gripper_cmd)}

    @property
    def is_teleop_active(self) -> bool:
        return self._teleop_active

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _tcp_actual_xyzrpy(self) -> list:
        rotvec = self.robot.get_tcp_pose().tolist()
        roll, pitch, yaw = self.tools.rotvec_to_rpy(rotvec[3:6])
        return [rotvec[0], rotvec[1], rotvec[2], roll, pitch, yaw]

    def _adjust_pika_to_arm(self, x, y, z, rx, ry, rz):
        """Apply the rigid offset between the pika frame (tracker output)
        and the arm tool frame so deltas line up with the wrist orientation."""
        T = self.tools.xyzrpy2Mat(x, y, z, rx, ry, rz)
        adj = self.tools.xyzrpy2Mat(*self.pika_to_arm)
        T = T @ adj
        return self.tools.mat2xyzrpy(T)

    def _refresh_tracker_pose(self):
        """Read the latest tracker pose, EMA-smooth it, then project into the
        arm frame. Smoothing happens BEFORE the increment computation so the
        damped signal flows through both servoL and the recorded action.
        """
        pose = self.sense.get_tracker_pose()
        if pose is None:
            return
        position, rotation = pose
        # Sense provides quaternion as [x, y, z, w]
        roll, pitch, yaw = self.tools.quaternion_to_rpy(*rotation)
        raw = np.array([position[0], position[1], position[2], roll, pitch, yaw],
                       dtype=float)

        # EMA on the raw tracker pose. We unwrap rotation deltas to the
        # nearest 2π so the low-pass doesn't get confused near the ±π wrap
        # of yaw/roll/pitch.
        if self.smoothing_alpha < 1.0 and self._smoothed_xyzrpy is not None:
            unwrapped = raw.copy()
            unwrapped[3:6] = self._smoothed_xyzrpy[3:6] + (
                ((raw[3:6] - self._smoothed_xyzrpy[3:6] + np.pi) % (2 * np.pi)) - np.pi
            )
            self._smoothed_xyzrpy = (
                self.smoothing_alpha * unwrapped
                + (1.0 - self.smoothing_alpha) * self._smoothed_xyzrpy
            )
        else:
            self._smoothed_xyzrpy = raw

        sx, sy, sz, sr, sp, syaw = self._smoothed_xyzrpy
        self._tracker_xyzrpy = list(self._adjust_pika_to_arm(
            sx, sy, sz, sr, sp, syaw,
        ))

    def _handle_trigger(self):
        """Toggle teleop mode on press, reset reference frame on release."""
        try:
            current = self.sense.get_command_state()
        except Exception:
            return
        if self._last_trigger is None:
            self._last_trigger = current
            return
        if current == self._last_trigger:
            return
        self._last_trigger = current

        self._teleop_active = not self._teleop_active
        if self._teleop_active:
            # Lock in the tracker pose at the moment the trigger was pressed.
            if self._tracker_xyzrpy is not None:
                self._base_pose = list(self._tracker_xyzrpy)
            print("[Teleop] >> ENGAGED")
        else:
            # On release, sync arm zero to the *current* TCP pose so the next
            # press doesn't snap the arm back.
            self._initial_pose_rpy = self._tcp_actual_xyzrpy()
            self._base_pose = list(self._initial_pose_rpy)
            print("[Teleop] << RELEASED")

    def _filter_target(self, target_xyzrpy: list) -> tuple[Optional[list], str]:
        """Apply the three-layer safety filter to a TCP-target candidate.

        Returns (clamped_xyzrpy_or_None, reason).
        - clamped_xyzrpy is the target after workspace clamping; None means
          the candidate was rejected outright (joint limits / tilt).
        - reason is empty string if accepted, else a short tag for logging.
        """
        out = list(target_xyzrpy)

        # 1) Workspace bbox clamp — soft, never reject, just clip into box.
        for axis_idx, axis_name in ((0, "x"), (1, "y"), (2, "z")):
            bounds = self.workspace_bounds.get(axis_name)
            if bounds and len(bounds) == 2 and bounds[0] is not None:
                out[axis_idx] = max(bounds[0], min(bounds[1], out[axis_idx]))

        # 2) Tilt cone — hard reject if gripper z-axis tilts too far from
        #    -world Z (i.e. "pointing up" or sideways past the cone).
        if self.max_tilt_from_down_rad is not None:
            T = self.tools.xyzrpy2Mat(*out)
            # Tool z-axis in world frame is the third column of the rotation.
            tool_z = np.array([T[0, 2], T[1, 2], T[2, 2]])
            # We want -world-Z aligned with the gripper's z (gripper points down).
            # cos(angle) = tool_z · (-z_world) = -tool_z[2]
            cos_angle = float(np.clip(-tool_z[2], -1.0, 1.0))
            tilt = float(np.arccos(cos_angle))
            if tilt > self.max_tilt_from_down_rad:
                return None, f"tilt={np.degrees(tilt):.0f}deg"

        # 3) Joint limits via UR's own IK — hard reject if any joint
        #    falls outside its software bound.
        if self.joint_limits is not None:
            try:
                rotvec = self.tools.rpy_to_rotvec(out[3], out[4], out[5])
                pose_for_ik = [out[0], out[1], out[2],
                               float(rotvec[0]), float(rotvec[1]),
                               float(rotvec[2])]
                # Use current Q as the "near" seed so IK picks the closest
                # branch (no joint flips).
                rtde_c = self.robot._rtde_c
                q_near = self.robot.get_state()["joint_positions"].tolist()
                # ur_rtde signature: getInverseKinematics(x, qnear,
                #   max_position_error=1e-10, max_orientation_error=1e-10)
                q_pred = rtde_c.getInverseKinematics(pose_for_ik, q_near)
                if q_pred and len(q_pred) == 6:
                    for i, qi in enumerate(q_pred):
                        bounds = self.joint_limits[i]
                        if not bounds:
                            continue
                        lo, hi = bounds
                        if lo is not None and qi < lo:
                            return None, f"q[{i}]={qi:.2f}<{lo}"
                        if hi is not None and qi > hi:
                            return None, f"q[{i}]={qi:.2f}>{hi}"
            except Exception:
                # IK call can fail on unreachable poses — treat as unreachable.
                return None, "IK_failed"

        return out, ""

    def _calc_pose_increment(self) -> Optional[list]:
        """Compute the next arm TCP target (xyzrpy) from current tracker reading."""
        if (self._tracker_xyzrpy is None
                or self._base_pose is None
                or self._initial_pose_rpy is None):
            return None

        begin = self.tools.xyzrpy2Mat(*self._base_pose)
        zero = self.tools.xyzrpy2Mat(*self._initial_pose_rpy)
        end = self.tools.xyzrpy2Mat(*self._tracker_xyzrpy)

        delta = np.linalg.inv(begin) @ end
        delta = np.array(delta, dtype=float)
        delta[:3, 3] *= self.position_scale
        result = zero @ delta
        return self.tools.mat2xyzrpy(result)

    def _loop(self):
        last_log = 0.0
        last_health = 0.0
        while self._running:
            t0 = time.time()

            # Health check at 2 Hz: every subsystem that can fail silently
            # gets probed. First subsystem that's dead aborts the loop with
            # a tagged reason so the wrapping script can print a clear
            # diagnosis instead of letting the user guess.
            if t0 - last_health > 0.5:
                last_health = t0
                health = [
                    ("UR RTDE control script",
                     self.robot.is_control_alive(),
                     "Protective Stop / Local mode / E-stop on the pendant"),
                    ("Pika Sense USB serial",
                     self.sense.is_alive(),
                     "USB cable unplugged, hub power loss, or Sense rebooted"),
                    ("Pika Gripper USB serial",
                     self.gripper.is_alive(),
                     "USB cable unplugged, 24V supply tripped, or gripper rebooted"),
                ]
                for name, ok, hint in health:
                    if not ok and not self.aborted:
                        self.aborted = True
                        self.controller_lost = True  # legacy alias
                        self.abort_reason = (
                            f"{name} dropped — likely cause: {hint}")
                        print(f"\n[Teleop] !!! {self.abort_reason}")
                        print("[Teleop] Auto-stopping. Fix the hardware, "
                              "then restart the script.")
                        self._running = False
                        break
                if not self._running:
                    break

            self._handle_trigger()
            self._refresh_tracker_pose()

            # Stream gripper command from the Sense encoder (always — the
            # operator might want to grasp without arm motion). EMA-smoothed
            # so micro-finger-tremor doesn't translate into gripper chatter.
            raw_gripper = self.sense.get_encoder_rad()
            if (self.gripper_smoothing_alpha < 1.0
                    and self._smoothed_gripper is not None):
                gripper_cmd = (
                    self.gripper_smoothing_alpha * raw_gripper
                    + (1.0 - self.gripper_smoothing_alpha) * self._smoothed_gripper
                )
            else:
                gripper_cmd = raw_gripper
            self._smoothed_gripper = gripper_cmd
            self.gripper.set_motor_angle(gripper_cmd)
            with self._lock:
                self._last_gripper_cmd = gripper_cmd

            # Compute and dispatch the next servoL target.
            target = self._calc_pose_increment()
            if target is not None and self._teleop_active:
                init_xyz = np.asarray(self._initial_pose_rpy[:3])
                tgt_xyz = np.asarray(target[:3])
                delta_m = float(np.linalg.norm(tgt_xyz - init_xyz))

                if delta_m > self.max_delta_m:
                    if time.time() - last_log > 0.2:
                        logger.warning(
                            "Hold pose: |Δ|=%.0fcm > max %.0fcm",
                            delta_m * 100, self.max_delta_m * 100,
                        )
                        last_log = time.time()
                    # Hold current pose to keep the RTDE watchdog fed.
                    try:
                        target = self._tcp_actual_xyzrpy()
                    except Exception:
                        target = None

            if target is not None and self._teleop_active:
                # Three-layer safety filter (workspace clamp + tilt + joint IK).
                filtered, reject_reason = self._filter_target(target)
                if filtered is None:
                    self._reject_count += 1
                    if time.time() - self._reject_log_t > 0.4:
                        logger.warning("Rejected servoL: %s (cumulative %d)",
                                       reject_reason, self._reject_count)
                        self._reject_log_t = time.time()
                    # IMPORTANT: do NOT just skip — ur_rtde's RTDEControl has
                    # a watchdog that requires servoL to be called continuously
                    # while teleop is engaged. If we skip too long the script
                    # is killed on the controller side ("RTDE control script
                    # is not running!"). Send the CURRENT pose instead so the
                    # arm stays put but the watchdog stays fed.
                    try:
                        target = self._tcp_actual_xyzrpy()
                    except Exception:
                        target = None
                else:
                    target = filtered

            if target is not None and self._teleop_active:
                rotvec = self.tools.rpy_to_rotvec(target[3], target[4], target[5])
                pose_cmd = [target[0], target[1], target[2],
                            float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]
                try:
                    self.robot.servo_l(
                        pose=pose_cmd,
                        speed=0.5,
                        acc=0.5,
                        dt=self.dt,
                        lookahead=0.1,
                        gain=300,
                    )
                    self._servo_fail_streak = 0
                except Exception as e:
                    self._servo_fail_streak += 1
                    if time.time() - last_log > 0.5:
                        logger.error("servoL failed: %s", e)
                        last_log = time.time()
                    if (self._servo_fail_streak >= self._servo_fail_limit
                            and not self.controller_lost):
                        self.controller_lost = True
                        print("\n[Teleop] !!! UR control dropped — "
                              "RTDE control script no longer running.")

            elapsed = time.time() - t0
            sleep = self.dt - elapsed
            if sleep > 0:
                time.sleep(sleep)


# ======================================================================
# Collector — episode + writer orchestration
# ======================================================================

class PikaCollector:
    def __init__(self, cfg: dict, writer: LeRobotWriter):
        self.cfg = cfg
        self.writer = writer

        # ----------------------- robot -----------------------
        self.robot = UR7eInterface(
            host=cfg["robot"]["host"],
            frequency=cfg["robot"].get("frequency", 500.0),
        )

        # ----------------------- sense -----------------------
        sense_cfg = cfg.get("pika_sense", {})
        self.sense = PikaSense(
            port=sense_cfg.get("port", "") or "",
            tracker_device=sense_cfg.get("tracker_device", "T20"),
            tracker_config=sense_cfg.get("tracker_config"),
            tracker_lh_config=sense_cfg.get("tracker_lh_config"),
        )

        # ----------------------- gripper ---------------------
        gripper_cfg = cfg.get("pika_gripper", {})
        wrist_cam = next((c for c in cfg.get("cameras", [])
                          if c.get("type") == "pika_wrist"
                          or c.get("source") == "pika_gripper"), None)
        self._wrist_cam_key = wrist_cam["name"] if wrist_cam else None
        self.gripper = PikaGripper(
            port=gripper_cfg.get("port", "") or "",
            wrist_camera_kind=(wrist_cam.get("kind", "realsense")
                               if wrist_cam else "none"),
            wrist_realsense_serial=(wrist_cam.get("serial")
                                    if wrist_cam else None),
            wrist_fisheye_index=(wrist_cam.get("device_index", 0)
                                 if wrist_cam else 0),
            wrist_width=(wrist_cam.get("width", 640) if wrist_cam else 640),
            wrist_height=(wrist_cam.get("height", 480) if wrist_cam else 480),
            wrist_fps=(wrist_cam.get("fps", 30) if wrist_cam else 30),
            enable_motor_on_connect=gripper_cfg.get("enable_motor", True),
        )

        # ----------------------- external cameras ------------
        external = [c for c in cfg.get("cameras", [])
                    if c.get("type") != "pika_wrist"
                    and c.get("source") != "pika_gripper"]
        self.ext_cameras = MultiCamera(external) if external else None
        self._ext_cam_keys = [c["name"] for c in external]

        # ----------------------- teleop controller -----------
        teleop_cfg = cfg.get("teleoperation", {})
        # PIKA_SCALE / PIKA_MAX_DELTA_M / PIKA_SMOOTHING env vars override
        # config — handy for tuning without editing yaml.
        position_scale = float(os.environ.get(
            "PIKA_SCALE", teleop_cfg.get("position_scale", 1.0)))
        max_delta_m = float(os.environ.get(
            "PIKA_MAX_DELTA_M", teleop_cfg.get("max_delta_m", 1.0)))
        smoothing_cfg = teleop_cfg.get("smoothing", {}) or {}
        smoothing_alpha = float(os.environ.get(
            "PIKA_SMOOTHING_ALPHA",
            smoothing_cfg.get("pose_alpha", 1.0)))
        gripper_smoothing_alpha = float(os.environ.get(
            "PIKA_GRIPPER_SMOOTHING_ALPHA",
            smoothing_cfg.get("gripper_alpha", 1.0)))
        safety_cfg = teleop_cfg.get("safety", {}) or {}
        max_tilt_deg = safety_cfg.get("max_tilt_from_down_deg")
        max_tilt_rad = (None if max_tilt_deg is None
                        else float(max_tilt_deg) * np.pi / 180.0)
        self.teleop = PikaTeleopController(
            robot=self.robot,
            sense=self.sense,
            gripper=self.gripper,
            pika_to_arm=teleop_cfg.get(
                "pika_to_arm",
                [0.0, 0.0, 0.0, 1.703151, 1.539109, 1.728148],
            ),
            position_scale=position_scale,
            max_delta_m=max_delta_m,
            servo_hz=int(teleop_cfg.get("servo_hz", 50)),
            smoothing_alpha=smoothing_alpha,
            gripper_smoothing_alpha=gripper_smoothing_alpha,
            workspace_bounds=safety_cfg.get("workspace") or {},
            joint_limits=safety_cfg.get("joint_limits"),
            max_tilt_from_down_rad=max_tilt_rad,
        )

        # Live preview window (cam_global + cam_wrist). Created in connect()
        # after the cameras are open, started/stopped around episodes so the
        # operator always sees what the robot sees.
        self.previewer: Optional[CameraPreviewer] = None
        preview_cfg = cfg.get("preview", {}) or {}
        self._enable_preview = bool(preview_cfg.get("enabled", True))
        self._preview_height = int(preview_cfg.get("height", 720))

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect(self):
        sense_port_pref = self.cfg.get("pika_sense", {}).get("port") or ""
        gripper_port_pref = self.cfg.get("pika_gripper", {}).get("port") or ""
        if not sense_port_pref or not gripper_port_pref:
            sp, gp = detect_pika_ports(sense_port_pref or None,
                                        gripper_port_pref or None)
            if not sense_port_pref:
                self.sense.port = sp
            if not gripper_port_pref:
                self.gripper.port = gp
            print(f"[Collector] Auto-detected ports — sense={self.sense.port}, "
                  f"gripper={self.gripper.port}")

        self.robot.connect(use_control=True)
        self.sense.connect()
        self.gripper.connect()
        if self.ext_cameras is not None:
            self.ext_cameras.connect()
        print("[Collector] All Pika devices ready.\n")

        if self._enable_preview:
            self.previewer = CameraPreviewer(
                frame_provider=self._capture_images,
                title="Pika cameras (q to close window)",
                fps=30,
                target_height=self._preview_height,
            )
            self.previewer.start()

    def disconnect(self):
        try:
            if self.previewer is not None:
                self.previewer.stop()
        except Exception:
            pass
        try:
            self.teleop.stop()
        except Exception:
            pass
        try:
            self.robot.stop()
        except Exception:
            pass
        self.robot.disconnect()
        self.sense.disconnect()
        self.gripper.disconnect()
        if self.ext_cameras is not None:
            self.ext_cameras.disconnect()

    # ------------------------------------------------------------------
    # frame composition
    # ------------------------------------------------------------------

    def _capture_images(self) -> Dict[str, np.ndarray]:
        images: Dict[str, np.ndarray] = {}
        if self.ext_cameras is not None:
            images.update(self.ext_cameras.get_latest_frames())
        if self._wrist_cam_key is not None:
            frame = self.gripper.get_wrist_frame()
            if frame is not None:
                images[self._wrist_cam_key] = frame
        return images

    def _build_state_action(self) -> tuple[np.ndarray, np.ndarray]:
        robot_state = self.robot.get_state()
        gripper_actual = self.gripper.get_motor_position()
        target_q = self.robot.get_target_q()
        gripper_cmd = self.teleop.get_command_snapshot()["gripper_cmd"]

        state = np.concatenate([
            robot_state["joint_positions"],
            np.array([gripper_actual], dtype=np.float32),
        ]).astype(np.float32)
        action = np.concatenate([
            target_q,
            np.array([gripper_cmd], dtype=np.float32),
        ]).astype(np.float32)
        return state, action

    # ------------------------------------------------------------------
    # episode loop
    # ------------------------------------------------------------------

    def run_episode(self, task: str) -> bool:
        print(f"\n{'='*60}")
        print(f"  Episode {self.writer.episode_index}  |  task: {task}")
        print(f"{'='*60}")
        print("  Pull the Pika Sense trigger to ENGAGE teleop, then release "
              "when ready.")
        print("  Press Enter here to START recording (Ctrl+C here = quit "
              "without recording this episode).")
        try:
            input(">> ")
        except (KeyboardInterrupt, EOFError):
            print("\n[Collector] Aborted before recording started.")
            return False

        fps = self.writer.fps
        period = 1.0 / fps
        self.writer.start_episode(task)
        if self.teleop._thread is None or not self.teleop._thread.is_alive():
            self.teleop.start()

        print(f"[Collector] Recording @ {fps} fps. Ctrl+C to stop.\n")
        start = time.time()
        n = 0
        controller_lost_during_episode = False
        try:
            while True:
                if self.teleop.aborted:
                    controller_lost_during_episode = True
                    print(f"\n[Collector] ABORTED — not user-initiated.")
                    print(f"[Collector] Reason: {self.teleop.abort_reason}")
                    break
                t0 = time.time()
                state, action = self._build_state_action()
                images = self._capture_images()
                self.writer.add_frame(
                    state=state,
                    action=action,
                    images=images,
                    timestamp=time.time() - start,
                )
                n += 1
                if self.previewer is not None:
                    state_lbl = "ENGAGED" if self.teleop.is_teleop_active else "released"
                    self.previewer.set_status(
                        f"REC ep{self.writer.episode_index}  frame {n}  "
                        f"{n / fps:.1f}s  teleop={state_lbl}"
                    )
                elapsed = time.time() - t0
                sleep = period - elapsed
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass
        if self.previewer is not None:
            self.previewer.set_status("")

        if self.writer._rows:
            self.writer._rows[-1]["next.done"] = True
        print(f"\n[Collector] Captured {n} frames ({n / fps:.1f}s).")

        # If the UR died mid-episode, force-discard — the trailing frames
        # are post-fault garbage that would poison the dataset.
        if controller_lost_during_episode:
            print("[Collector] Auto-discarding episode (controller dropped).")
            self.writer.end_episode(discard=True)
            return False

        try:
            choice = input("Save this episode? [Y/n/q] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            choice = "q"
            print()
        if choice == "q":
            self.writer.end_episode(discard=True)
            return False
        self.writer.end_episode(discard=(choice == "n"))
        return True

    def run(self, tasks):
        """Run the collection loop. ``tasks`` is a list of task strings;
        the operator picks one per episode by typing its number.
        """
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = list(tasks)

        def _menu():
            print("\nAvailable tasks:")
            for i, t in enumerate(tasks, 1):
                print(f"  [{i}] {t}")
            print("  [a] add a new task on the fly")
            print("  [s] start episode with last-used task "
                  f"({tasks[self._last_task_idx]!r})")
            print("  [q] quit")

        try:
            self.teleop.start()
            self._last_task_idx = 0
            while True:
                _menu()
                try:
                    cmd = input(">> ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print()
                    break
                if cmd == "q":
                    break
                if cmd == "a":
                    try:
                        new_task = input("New task instruction: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        print()
                        continue
                    if new_task:
                        tasks.append(new_task)
                        self._last_task_idx = len(tasks) - 1
                        print(f"  Added task [{len(tasks)}]: {new_task!r}")
                    continue
                if cmd == "s" or cmd == "":
                    task = tasks[self._last_task_idx]
                elif cmd.isdigit():
                    idx = int(cmd) - 1
                    if not (0 <= idx < len(tasks)):
                        print(f"  !! Invalid task number: {cmd}")
                        continue
                    self._last_task_idx = idx
                    task = tasks[idx]
                else:
                    print(f"  !! Unknown command: {cmd!r}")
                    continue
                if not self.run_episode(task):
                    break
        finally:
            self.writer.finalize()
            self.disconnect()


# ======================================================================
# Entry
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Pika teleoperation → LeRobot v3.0 dataset collection")
    p.add_argument("--config", default="configs/pika_config.yaml")
    p.add_argument("--output_dir", default=None,
                   help="Override collection.output_dir from the config.")
    p.add_argument("--dataset_name", default=None,
                   help="Override collection.dataset_name from the config.")
    # Multi-task input. Either a single instruction or a comma-separated list:
    #   --task "pick red block and place on tray"
    #   --tasks "pick red block,place on tray,pick blue cube"
    # Or a file with one instruction per line:
    #   --tasks_file tasks.txt
    p.add_argument("--task", default=None,
                   help="Single task instruction (legacy single-task mode).")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated list of task instructions; pick one "
                        "per episode at the prompt.")
    p.add_argument("--tasks_file", default=None,
                   help="Path to a UTF-8 text file with one task per line.")
    p.add_argument("--fps", type=int, default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    fps = args.fps or cfg["collection"].get("fps", 30)
    output_dir = args.output_dir or cfg["collection"].get("output_dir", "datasets")
    dataset_name = args.dataset_name or cfg["collection"].get(
        "dataset_name", "ur7e_pika_demo")

    cam_keys = [c["name"] for c in cfg.get("cameras", [])]
    image_size = (
        cfg["cameras"][0].get("height", 480) if cam_keys else 480,
        cfg["cameras"][0].get("width", 640) if cam_keys else 640,
    )

    writer = LeRobotWriter(
        output_dir=output_dir,
        dataset_name=dataset_name,
        fps=fps,
        camera_keys=cam_keys,
        state_dim=len(STATE_NAMES),
        action_dim=len(ACTION_NAMES),
        state_names=STATE_NAMES,
        action_names=ACTION_NAMES,
        robot_type="ur7e",
        image_size=image_size,
        action_is_commanded=True,  # action = commanded targetQ + gripper_cmd
    )

    # Resolve task list — priority: --tasks_file  >  --tasks  >  --task
    tasks: list[str] = []
    if args.tasks_file:
        with open(args.tasks_file, encoding="utf-8") as f:
            tasks = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    elif args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    elif args.task:
        tasks = [args.task]
    else:
        tasks = ["manipulation"]
    print(f"[Collector] {len(tasks)} task(s) loaded:")
    for i, t in enumerate(tasks, 1):
        print(f"  [{i}] {t}")

    collector = PikaCollector(cfg, writer)
    collector.connect()
    collector.run(tasks=tasks)


if __name__ == "__main__":
    main()
