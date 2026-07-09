# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


def object_position_in_world_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Observation the position of the object in the world frame."""
    object = env.scene[asset_cfg.name]
    return wp.to_torch(object.data.root_pos_w)


def object_position_in_frame(
    env: ManagerBasedRLEnv,
    root_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    root_frame: RigidObject = env.scene[root_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = wp.to_torch(object.data.root_pos_w)[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        wp.to_torch(root_frame.data.root_pos_w), wp.to_torch(root_frame.data.root_quat_w), object_pos_w
    )
    return object_pos_b


def object_angular_velocity_norm(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """L2 norm of a rigid object's root angular velocity in the world frame (rad/s)."""
    object: RigidObject = env.scene[object_cfg.name]
    ang_vel_w = wp.to_torch(object.data.root_ang_vel_w)
    return torch.norm(ang_vel_w, dim=1, keepdim=True)
