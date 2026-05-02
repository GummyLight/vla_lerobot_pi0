"""lerobot-aligned robot drivers for this project.

Importing this subpackage triggers `RobotConfig.register_subclass("ur7e_follower")`
via the side-effecting import of `ur7e_follower.config_ur7e_follower`. After
that, `make_robot_from_config(UR7eFollowerConfig(...))` (and the equivalent
draccus-driven `--robot.type=ur7e_follower ...` CLI) just works.
"""

from __future__ import annotations

from .ur7e_follower import UR7eFollower, UR7eFollowerConfig  # noqa: F401

__all__ = ["UR7eFollower", "UR7eFollowerConfig"]
