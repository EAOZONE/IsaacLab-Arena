# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cartesian-space waypoint playback for scripted (non-teleop) end-effector motion.

Builds per-step ``(position, quaternion, hand)`` targets by lerping/slerping
between waypoints. All tensors are single poses ``(3,)``/``(4,)`` (quaternions
in ``x, y, z, w`` order); this module does not batch over environments.
"""

from __future__ import annotations

import torch
from collections.abc import Iterator
from dataclasses import dataclass

from isaaclab.utils.math import quat_slerp


@dataclass
class LinearSegment:
    """Straight-line position lerp + quaternion slerp from pose0 to pose1."""

    pos0: torch.Tensor
    quat0: torch.Tensor
    hand0: torch.Tensor
    pos1: torch.Tensor
    quat1: torch.Tensor
    hand1: torch.Tensor
    steps: int


Segment = LinearSegment


def play_segments(segments: list[Segment]) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield ``(pos, quat, hand)`` for every step of every segment, in order.

    Each segment yields exactly ``segment.steps`` poses, with the interpolation
    parameter sweeping from ``1/steps`` to ``1.0``. Segments are meant to be
    chained end-to-end (each one's ``pos0``/``quat0``/``hand0`` should match
    the previous segment's endpoint) -- the ``tau=0`` pose is never re-yielded.
    """
    for segment in segments:
        assert segment.steps > 0, f"segment must have at least 1 step, got {segment.steps}"
        for step in range(1, segment.steps + 1):
            tau = step / segment.steps
            hand = torch.lerp(segment.hand0, segment.hand1, tau)
            pos = torch.lerp(segment.pos0, segment.pos1, tau)
            quat = quat_slerp(segment.quat0, segment.quat1, tau)
            yield pos, quat, hand


def total_steps(segments: list[Segment]) -> int:
    return sum(segment.steps for segment in segments)
