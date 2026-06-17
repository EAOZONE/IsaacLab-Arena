# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Spawn an arbitrary articulation USD with an :class:`~isaaclab_arena.affordances.openable.Openable` joint."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose


class OpenableArticulation(Object, Openable):
    """Articulation asset from a user-supplied USD path and revolute/prismatic joint name."""

    def __init__(
        self,
        name: str,
        usd_path: str,
        openable_joint_name: str,
        openable_threshold: float = 0.5,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        tags: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            tags=tags or ["object", "openable"],
            prim_path=prim_path,
            usd_path=usd_path,
            object_type=ObjectType.ARTICULATION,
            scale=scale,
            initial_pose=initial_pose,
            openable_joint_name=openable_joint_name,
            openable_threshold=openable_threshold,
            **kwargs,
        )
