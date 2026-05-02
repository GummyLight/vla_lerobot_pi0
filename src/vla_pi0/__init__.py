"""vla_pi0 — lerobot-aligned hardware + script layer for the UR7e + Robotiq + 2× D435i rig.

This package mirrors the upstream `lerobot` layout (`robots/`, `cameras/`,
`configs/`, `scripts/`) so that a UR7e + Robotiq 2F-58 + dual RealSense D435i
setup can be plugged into any lerobot-style entrypoint.

Importing the top-level package eagerly registers the `ur7e_follower` robot
subclass with `lerobot.robots.config.RobotConfig`, so downstream code only
needs `import vla_pi0` (or `import vla_pi0.robots`) before constructing
configs by string name.
"""

from __future__ import annotations

from . import robots  # noqa: F401 — side-effect: register ur7e_follower

__all__ = ["robots"]
