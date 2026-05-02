"""UR7e + Robotiq 2F-58 follower, packaged as a lerobot Robot."""

from __future__ import annotations

from .config_ur7e_follower import UR7eFollowerConfig
from .ur7e_follower import UR7eFollower

__all__ = ["UR7eFollower", "UR7eFollowerConfig"]
