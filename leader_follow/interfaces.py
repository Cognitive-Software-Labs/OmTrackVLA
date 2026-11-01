"""Data passed between the leader-following stages.

Frames: image boxes are pixel ``xyxy``; metric positions are metres in the
odometry frame (``odom``) unless a field says otherwise.  Each stage consumes
the previous stage's type, so any stage can be swapped (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

# SEARCHING: no leader yet. LOCKED: leader identity confirmed; the only state that
# may drive motion. UNCERTAIN: leader not confirmed this frame. LOST: unconfirmed
# for long enough that the robot must stop and search.
LeaderState = Literal["SEARCHING", "LOCKED", "UNCERTAIN", "LOST"]


@dataclass(frozen=True)
class CameraTrack:
    """One person tracked in the camera image."""
    track_id: int
    bbox: np.ndarray        # xyxy pixels, float32
    score: float            # detector confidence
    feature: np.ndarray     # appearance embedding used for identity (L2-normalised)


@dataclass(frozen=True)
class LidarPerson:
    """One person detected in the merged 360° scan."""
    position: Tuple[float, float]   # x, y in base_link, metres
    score: float


@dataclass(frozen=True)
class PersonTrack:
    """One person after lidar-camera fusion."""
    track_id: int
    position: Tuple[float, float]   # odom, metres
    velocity: Tuple[float, float]   # odom, metres/second
    feature: Optional[np.ndarray]   # latest appearance embedding; None while out of camera view


@dataclass(frozen=True)
class LeaderEstimate:
    state: LeaderState
    reason: str
    track_id: Optional[int]
    position: Optional[Tuple[float, float]]   # odom, metres; None unless a leader track exists


@dataclass(frozen=True)
class FollowCommand:
    """Body-frame velocity for the omnidirectional base (geometry_msgs/Twist)."""
    vx: float   # m/s forward
    vy: float   # m/s left
    wz: float   # rad/s counter-clockwise
