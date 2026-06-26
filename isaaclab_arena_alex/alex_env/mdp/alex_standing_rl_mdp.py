# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""MDP helpers for Alex in-place standing balance RL."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_STANDING_FULL_JOINT_POS,
    ALEX_UPPER_BODY_NUBS_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ARM_NOISE_BUFFER_KEY = "_alex_standing_arm_noise"


def _get_arm_noise_buffer(env: ManagerBasedRLEnv, num_arm_joints: int) -> torch.Tensor:
    buffer = getattr(env, _ARM_NOISE_BUFFER_KEY, None)
    if buffer is None or buffer.shape != (env.num_envs, num_arm_joints):
        buffer = torch.zeros(env.num_envs, num_arm_joints, device=env.device)
        setattr(env, _ARM_NOISE_BUFFER_KEY, buffer)
    return buffer


def reset_alex_standing_arm_noise(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Zero arm disturbance offsets on reset."""
    del asset_cfg
    if env_ids is None:
        env_ids = slice(None)
    buffer = getattr(env, _ARM_NOISE_BUFFER_KEY, None)
    if buffer is not None:
        buffer[env_ids] = 0.0


def resample_alex_standing_arm_noise(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    noise_std: float,
) -> None:
    """Sample new random arm joint offsets to emulate reaching disturbances."""
    del asset_cfg
    if env_ids is None:
        env_ids = slice(None)
    num_resets = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
    num_arm_joints = len(ALEX_UPPER_BODY_NUBS_JOINT_NAMES)
    buffer = _get_arm_noise_buffer(env, num_arm_joints)
    buffer[env_ids] = torch.randn((num_resets, num_arm_joints), device=env.device) * noise_std


def hold_alex_upper_body_joints(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    nominal_positions: dict[str, float] | None = None,
) -> None:
    """Command the upper body to a nominal pose plus any active disturbance offsets."""
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)

    joint_ids, joint_names = robot.find_joints(list(ALEX_UPPER_BODY_NUBS_JOINT_NAMES), preserve_order=True)
    nominal = nominal_positions or ALEX_STANDING_FULL_JOINT_POS
    default_joint_pos = wp.to_torch(robot.data.default_joint_pos)
    targets = default_joint_pos[:, joint_ids].clone()
    for idx, name in enumerate(joint_names):
        if name in nominal:
            targets[:, idx] = nominal[name]
    noise = _get_arm_noise_buffer(env, len(joint_names))
    targets = targets + noise
    robot.set_joint_position_target(targets, joint_ids)


def reset_alex_standing_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    joint_positions: dict[str, float] | None = None,
) -> None:
    """Reset selected environments to the nominal standing joint configuration."""
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    pose = joint_positions or ALEX_STANDING_FULL_JOINT_POS

    joint_pos = wp.to_torch(robot.data.default_joint_pos).clone()
    joint_vel = torch.zeros_like(joint_pos)
    for name, value in pose.items():
        joint_ids, _ = robot.find_joints(name)
        joint_pos[env_ids, joint_ids] = value
        joint_vel[env_ids, joint_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)


def alex_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal base linear velocity using an L2 squared kernel."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(wp.to_torch(robot.data.root_lin_vel_b)[:, :2]), dim=1)


def alex_bad_orientation(
    env: ManagerBasedRLEnv,
    limit_xy: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the pelvis tilt exceeds a projected-gravity threshold."""
    robot: Articulation = env.scene[asset_cfg.name]
    tilt = torch.linalg.vector_norm(wp.to_torch(robot.data.projected_gravity_b)[:, :2], dim=1)
    return tilt > limit_xy


def alex_base_diverged(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 10.0,
    max_height: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base goes physically unstable (flies away / explodes).

    The fall terminations (``root_height_below_minimum``, tilt) only catch a robot
    that drops or topples. A floating base can instead diverge numerically by flying
    upward or translating at huge velocity while staying upright, which never trips
    those checks and lets a single env accumulate near-inf penalties that poison the
    PPO value target. This guard resets such envs before they corrupt the batch. It
    also catches NaN/inf state (``> max`` is False for NaN, so test the finite case).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    lin_vel = wp.to_torch(robot.data.root_lin_vel_w)
    height = wp.to_torch(robot.data.root_pos_w)[:, 2]
    speed = torch.linalg.vector_norm(lin_vel, dim=1)
    nonfinite = ~torch.isfinite(speed) | ~torch.isfinite(height)
    return (speed > max_lin_vel) | (height > max_height) | nonfinite
