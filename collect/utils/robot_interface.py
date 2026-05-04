"""
UR7e robot interface via ur_rtde.

Install: pip install ur-rtde
"""

from __future__ import annotations

import re
import socket
import time
from typing import Dict, List, Optional

import numpy as np


URSCRIPT_INJECT_PORT = 30003   # Realtime interface — accepts URScript and stays open on PolyScope X.
                               # Primary (30001) and Secondary (30002) are sometimes locked down on
                               # newer firmware (UR7e/UR12e/UR15/UR20 PolyScope X); 30003 is the most
                               # reliable port for sending complete .script programs externally.


class UR7eInterface:
    """Thin wrapper around ur_rtde for state reading and motion control."""

    def __init__(self, host: str, frequency: float = 500.0):
        self.host = host
        self.frequency = frequency
        self._rtde_r = None
        self._rtde_c = None

    def connect(self, use_control: bool = True):
        """Connect RTDE interfaces.

        use_control=True  → opens RTDEReceive + RTDEControl (needed for
                            send_urscript / moveJ / servoL / freedrive ...)
        use_control=False → opens RTDEReceive only. Use this when you intend
                            to play full URScript programs via the primary
                            interface (port 30001); ur_rtde's control-loop
                            script would otherwise re-upload itself on top of
                            yours and prevent it from ever reaching motion.
        """
        import rtde_receive
        self._rtde_r = rtde_receive.RTDEReceiveInterface(self.host, self.frequency)
        if use_control:
            import rtde_control
            self._rtde_c = rtde_control.RTDEControlInterface(self.host)
        print(f"[Robot] Connected to UR7e @ {self.host} "
              f"(control={'on' if use_control else 'off'})")

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

    def get_target_q(self) -> np.ndarray:
        """Return the controller's commanded joint targets (for teleop action labels)."""
        return np.array(self._rtde_r.getTargetQ(), dtype=np.float32)

    def get_tcp_pose(self) -> np.ndarray:
        """Return the current actual TCP pose [x, y, z, rx, ry, rz] in metres + rotvec."""
        return np.array(self._rtde_r.getActualTCPPose(), dtype=np.float32)

    def is_control_alive(self) -> bool:
        """True iff the RTDEControl script is still running on the controller.

        ur_rtde's C++ watchdog spams ``RTDE control script is not running!`` to
        stderr when the program drops (Protective Stop, Local mode toggle,
        E-stop, etc.). servoL no longer raises in that state — it just returns
        ``False`` after the spam — so polling this flag is the only reliable
        Python-side detection.
        """
        if self._rtde_c is None:
            return False
        try:
            return bool(self._rtde_c.isProgramRunning())
        except Exception:
            return False

    def is_steady(self, vel_threshold: float = 0.01) -> bool:
        """Return True when all joints are nearly stationary."""
        return float(np.max(np.abs(self._rtde_r.getActualQd()))) < vel_threshold

    # ------------------------------------------------------------------
    # Motion control
    # ------------------------------------------------------------------

    def send_urscript(self, script: str):
        """Send a raw URScript snippet (e.g. a single `movej(...)`) via the
        ur_rtde control interface. Best for short, non-program-style scripts."""
        if self._rtde_c is None:
            raise RuntimeError(
                "send_urscript requires RTDEControl. Reconnect with "
                "use_control=True, or use play_program() for full PolyScope "
                "programs."
            )
        self._rtde_c.sendCustomScript(script)

    def _dashboard(self, cmd: str, timeout: float = 3.0) -> str:
        """Send a one-shot command to the Dashboard server (port 29999)."""
        with socket.create_connection((self.host, 29999), timeout=timeout) as s:
            s.recv(256)  # discard welcome banner
            s.sendall(cmd.encode("utf-8") + b"\n")
            return s.recv(512).decode("utf-8", errors="replace").strip()

    def play_program(self, script: str, timeout: float = 5.0,
                     post_send_sleep: float = 1.0):
        """Send a complete PolyScope-style URScript program to the Realtime
        interface (port 30003) and let the controller execute it as a
        top-level program.

        PolyScope-exported `.script` files wrap their entire body in
            def P1():
              ...
            end
        but never call `P1()` at the end — PolyScope adds the call implicitly
        on Play. We detect that pattern and append the call automatically so
        the program actually runs.

        Parameters
        ----------
        script : str
            URScript text. Either a function-style PolyScope export or a
            free-form sequence of top-level statements.
        timeout : float
            Socket connect timeout (seconds).
        post_send_sleep : float
            Delay after sending so the controller has time to compile and
            start the program before we start polling state. The Realtime
            interface buffers the script and may take ~0.5–1 s to begin.
            Set higher (~3 s) if your script has a long preamble of
            URCap installation calls.

        After `play_program` returns, the robot is running the script. Use
        `get_state()` (RTDEReceive only) to track progress; do not issue
        `RTDEControl` motion commands until the program ends.
        """
        text = script.strip()

        m = re.search(r"^def\s+([A-Za-z_]\w*)\s*\(\s*\)\s*:", text, re.MULTILINE)
        if m:
            fname = m.group(1)
            already_called = re.search(rf"^\s*{re.escape(fname)}\s*\(\s*\)\s*$",
                                       text, re.MULTILINE) is not None
            if not already_called:
                text = text + f"\n{fname}()\n"
                print(f"[Robot] Auto-appended call to top-level function '{fname}()'")

        if self._rtde_c is not None:
            print("[Robot] !! WARNING: RTDEControl is connected; its keep-alive "
                  "thread will re-upload its control script and silently kill "
                  "your program. Reconnect with use_control=False.")

        payload = (text + "\n").encode("utf-8")
        with socket.create_connection((self.host, URSCRIPT_INJECT_PORT),
                                      timeout=timeout) as s:
            s.sendall(payload)
        print(f"[Robot] Sent {len(payload)} bytes to {self.host}:{URSCRIPT_INJECT_PORT} "
              f"(realtime interface)")

        # Give the controller a moment to compile and start the program
        # before we poll state — otherwise dashboard reports running=false
        # purely because the script hasn't begun yet. Then poll repeatedly
        # so a slow-compiling 80 KB program isn't flagged as failed just
        # because our first check landed in the compile window.
        time.sleep(post_send_sleep)
        deadline = time.time() + 4.0
        last_reply = ""
        running = False
        while time.time() < deadline:
            try:
                last_reply = self._dashboard("running")
                if "true" in last_reply.lower():
                    running = True
                    break
            except Exception as e:
                last_reply = f"<dashboard error: {e}>"
            time.sleep(0.3)

        if running:
            print(f"[Robot] Dashboard: {last_reply}")
        else:
            print(f"[Robot] !! Program did NOT start within {time.time() - (deadline - 4.0):.1f}s.")
            print(f"           Last dashboard reply: {last_reply}")
            print("           Common causes:")
            print("           - PolyScope not in Remote Control mode")
            print("           - Joint jitter too large -> IK failure (try --joint_jitter 0.01)")
            print("           - URCap functions in the script not installed (rq_*, etc.)")
            print("           - Another program loaded with errors on the pendant")
            print("           Check the pendant Log tab for the controller's actual error.")

    def send_urscript_file(self, path: str):
        with open(path, encoding="utf-8") as f:
            script = f.read()
        self.play_program(script)

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

    def servo_j(
        self,
        q: List[float],
        speed: float = 0.5,
        acc: float = 0.5,
        dt: float = 0.002,
        lookahead: float = 0.1,
        gain: float = 300,
    ):
        """Streaming joint-space servo (use when you've already chosen the
        IK branch — e.g. base-biased teleop)."""
        self._rtde_c.servoJ(q, speed, acc, dt, lookahead, gain)

    def get_inverse_kinematics(
        self, pose_rotvec: List[float], q_near: Optional[List[float]] = None,
        max_pos_err: float = 1e-3, max_ori_err: float = 1e-3,
    ) -> Optional[List[float]]:
        """Resolve the closest IK solution to ``q_near`` (or current Q if
        omitted). Returns None if UR's solver cannot reach the pose.

        IMPORTANT: ur_rtde's default tolerances are 1e-10 — that's below
        floating-point precision for pose math, so the Newton-Raphson
        solver almost always fails to converge with the defaults and
        silently returns a non-solution. We pass engineering-grade
        tolerances (1e-3 m / 1e-3 rad ≈ 1mm / 0.06°) so the solver
        actually returns a usable joint configuration. UR's own internal
        servoL uses similar tolerances under the hood, so there's no
        precision loss vs. the native path.
        """
        if self._rtde_c is None:
            return None
        if q_near is None:
            q_near = self._rtde_r.getActualQ()
        try:
            sol = self._rtde_c.getInverseKinematics(
                pose_rotvec, q_near, max_pos_err, max_ori_err)
            if sol is None:
                return None
            sol = list(sol)
            return sol if len(sol) == 6 else None
        except Exception:
            return None

    def servo_stop(self):
        self._rtde_c.servoStop()

    def freedrive_mode(self, enable: bool):
        if self._rtde_c is None:
            raise RuntimeError("freedrive requires RTDEControl (use_control=True).")
        if enable:
            self._rtde_c.teachMode()
        else:
            self._rtde_c.endTeachMode()

    def stop(self, deceleration: float = 2.0):
        if self._rtde_c is None:
            return
        self._rtde_c.stopJ(deceleration)

    def is_program_running(self) -> bool:
        return self._rtde_r.isRobotMoving()
