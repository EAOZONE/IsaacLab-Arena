# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cartesian-space waypoint playback for scripted (non-teleop) end-effector motion.

Builds per-step ``(position, quaternion, hand)`` targets by interpolating
between waypoints -- either a straight-line pose interpolation, or a circular
arc about a known world-frame pivot/axis (for following a hinge or lever's
known rotation). All tensors are single poses ``(3,)``/``(4,)`` (quaternions
in ``x, y, z, w`` order); this module does not batch over environments.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, quat_slerp


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


@dataclass
class ArcSegment:
    """Rotate pose0 by ``angle_rad`` about ``axis`` through ``pivot`` (world frame).

    The hand target lerps from ``hand0`` to ``hand1`` over the arc (pass the
    same tensor for both to hold it steady, e.g. while pulling a lever with a
    closed grip).
    """

    pos0: torch.Tensor
    quat0: torch.Tensor
    hand0: torch.Tensor
    pivot: torch.Tensor
    axis: torch.Tensor
    angle_rad: float
    hand1: torch.Tensor
    steps: int


Segment = LinearSegment | ArcSegment


def arc_pose_at(segment: ArcSegment, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose reached after sweeping ``tau`` (0..1) of ``segment``'s arc."""
    angle = torch.tensor([segment.angle_rad * tau], device=segment.axis.device, dtype=segment.axis.dtype)
    step_rot = quat_from_angle_axis(angle, segment.axis.unsqueeze(0)).squeeze(0)
    pos = segment.pivot + quat_apply(step_rot.unsqueeze(0), (segment.pos0 - segment.pivot).unsqueeze(0)).squeeze(0)
    quat = quat_mul(step_rot.unsqueeze(0), segment.quat0.unsqueeze(0)).squeeze(0)
    return pos, quat


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
            if isinstance(segment, LinearSegment):
                pos = torch.lerp(segment.pos0, segment.pos1, tau)
                quat = quat_slerp(segment.quat0, segment.quat1, tau)
            elif isinstance(segment, ArcSegment):
                pos, quat = arc_pose_at(segment, tau)
            else:
                raise TypeError(f"Unknown segment type: {type(segment)}")
            yield pos, quat, hand


def total_steps(segments: list[Segment]) -> int:
    return sum(segment.steps for segment in segments)
