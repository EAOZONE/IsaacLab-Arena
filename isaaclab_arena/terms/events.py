# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import warp as wp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.velocity import Velocity


def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    velocity: Velocity | None = None,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # Convert the pose to the env frame
    pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_t_xyz_q_xyzw[:, :3] += env.scene.env_origins[env_ids]
    # Set the pose and velocity
    asset.write_root_pose_to_sim(pose_t_xyz_q_xyzw, env_ids=env_ids)
    if velocity is not None:
        vel = velocity.to_tensor(device=env.device).unsqueeze(0).expand(num_envs, -1)
        asset.write_root_velocity_to_sim(vel, env_ids=env_ids)
    else:
        asset.write_root_velocity_to_sim(
            torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids
        )


def set_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    if env_ids is None:
        return

    # Grab the object
    asset = env.scene[asset_cfg.name]

    # Set the objects pose in each environment independently
    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        # Convert the pose to the env frame
        pose = pose_list[cur_env]
        pose_t_xyz_q_xyzw = pose.to_tensor(device=env.device).unsqueeze(0)
        pose_t_xyz_q_xyzw[0, :3] += env.scene.env_origins[cur_env, :]
        # Set the pose and velocity
        asset.write_root_pose_to_sim(
            pose_t_xyz_q_xyzw, env_ids=torch.tensor([cur_env], device=env.device)
        )
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([cur_env], device=env.device),
        )


def reset_all_articulation_joints(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the articulation joints to the initial state."""
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = wp.to_torch(articulation_asset.data.default_root_state)[
            env_ids
        ].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(
            default_root_state[:, :7], env_ids=env_ids
        )
        articulation_asset.write_root_velocity_to_sim(
            default_root_state[:, 7:], env_ids=env_ids
        )
        # obtain default joint positions
        default_joint_pos = wp.to_torch(articulation_asset.data.default_joint_pos)[
            env_ids
        ].clone()
        default_joint_vel = wp.to_torch(articulation_asset.data.default_joint_vel)[
            env_ids
        ].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(
            default_joint_pos, default_joint_vel, env_ids=env_ids
        )


def reset_internal_rigid_body_to_object_rest(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str,
    body_suffix: str,
    object_pose: Pose,
    object_scale: tuple[float, float, float],
    body_local_pos: tuple[float, float, float],
    body_local_quat_xyzw: tuple[float, float, float, float],
) -> None:
    """Reset a rigid body nested inside a base USD asset back to its authored rest pose."""
    if env_ids is None:
        return

    object_pose_tensor = (
        object_pose.to_tensor(device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
    )
    object_pose_tensor[:, :3] += env.scene.env_origins[env_ids]
    _reset_internal_rigid_body_to_pose(
        env=env,
        env_ids=env_ids,
        object_name=object_name,
        body_suffix=body_suffix,
        object_pose_tensor=object_pose_tensor,
        object_scale=object_scale,
        body_local_pos=body_local_pos,
        body_local_quat_xyzw=body_local_quat_xyzw,
    )


def _reset_internal_rigid_body_to_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str,
    body_suffix: str,
    object_pose_tensor: torch.Tensor,
    object_scale: tuple[float, float, float],
    body_local_pos: tuple[float, float, float],
    body_local_quat_xyzw: tuple[float, float, float, float],
) -> None:
    """Reset a nested rigid body from an already env-origin-adjusted object pose tensor."""
    from isaaclab.utils.math import quat_apply, quat_mul
    from isaacsim.core.prims import RigidPrim

    object_pos = object_pose_tensor[:, :3]
    object_quat = object_pose_tensor[:, 3:]
    local_pos = torch.tensor(body_local_pos, device=env.device, dtype=object_pos.dtype)
    scale = torch.tensor(object_scale, device=env.device, dtype=object_pos.dtype)
    local_quat = torch.tensor(
        body_local_quat_xyzw, device=env.device, dtype=object_pos.dtype
    )
    local_quat = local_quat.unsqueeze(0).repeat(len(env_ids), 1)

    body_pos = object_pos + quat_apply(
        object_quat, local_pos.unsqueeze(0).repeat(len(env_ids), 1) * scale
    )
    body_quat_xyzw = quat_mul(object_quat, local_quat)
    body_quat_wxyz = body_quat_xyzw[:, [3, 0, 1, 2]]
    zeros = torch.zeros((1, 3), device=env.device, dtype=object_pos.dtype)

    if not hasattr(env, "_lever_rest_quat_by_object"):
        env._lever_rest_quat_by_object = {}
    rest_quats = getattr(env, "_lever_rest_quat_by_object").get(object_name)
    if rest_quats is None or rest_quats.shape[0] != env.num_envs:
        rest_quats = torch.zeros(
            (env.num_envs, 4), device=env.device, dtype=object_pos.dtype
        )
        rest_quats[:, 3] = 1.0
    rest_quats[env_ids] = body_quat_xyzw
    env._lever_rest_quat_by_object[object_name] = rest_quats

    for row, env_id in enumerate(env_ids.tolist()):
        body = RigidPrim(f"/World/envs/env_{env_id}/{object_name}{body_suffix}")
        body.set_world_poses(
            positions=body_pos[row : row + 1],
            orientations=body_quat_wxyz[row : row + 1],
        )
        body.set_linear_velocities(zeros)
        body.set_angular_velocities(zeros)


def _sample_root_pose_xy_yaw(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_position_xyz: tuple[float, float, float],
    base_yaw_rad: float,
    xy_jitter: float,
    yaw_jitter_rad: float,
    z_jitter: float = 0.0,
) -> torch.Tensor:
    """Return env-origin-adjusted root poses as ``xyz + xyzw`` tensors."""
    from isaaclab.utils.math import quat_from_euler_xyz

    num_envs = len(env_ids)
    base_pos = torch.tensor(
        base_position_xyz, device=env.device, dtype=torch.float32
    ).repeat(num_envs, 1)
    if xy_jitter > 0.0:
        base_pos[:, :2] += (
            torch.rand((num_envs, 2), device=env.device) * 2.0 - 1.0
        ) * xy_jitter
    if z_jitter > 0.0:
        base_pos[:, 2] += (
            torch.rand((num_envs,), device=env.device) * 2.0 - 1.0
        ) * z_jitter
    base_pos += env.scene.env_origins[env_ids]

    yaw = torch.full((num_envs,), base_yaw_rad, device=env.device, dtype=torch.float32)
    if yaw_jitter_rad > 0.0:
        yaw += (torch.rand((num_envs,), device=env.device) * 2.0 - 1.0) * yaw_jitter_rad
    quat_xyzw = quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw,
    )
    return torch.cat([base_pos, quat_xyzw], dim=-1)


def randomize_articulation_root_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    base_position_xyz: tuple[float, float, float],
    base_yaw_rad: float,
    xy_jitter: float,
    yaw_jitter_rad: float,
    z_jitter: float = 0.0,
) -> None:
    """Randomize an articulation root pose around a fixed xy/yaw center."""
    if env_ids is None:
        return

    asset = env.scene[asset_cfg.name]
    root_pose = _sample_root_pose_xy_yaw(
        env=env,
        env_ids=env_ids,
        base_position_xyz=base_position_xyz,
        base_yaw_rad=base_yaw_rad,
        xy_jitter=xy_jitter,
        z_jitter=z_jitter,
        yaw_jitter_rad=yaw_jitter_rad,
    )
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )


def randomize_base_lever_pose_and_reset_handle(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_name: str,
    body_suffix: str,
    base_position_xyz: tuple[float, float, float],
    base_yaw_rad: float,
    xy_jitter: float,
    z_jitter: float,
    yaw_jitter_rad: float,
    object_scale: tuple[float, float, float],
    body_local_pos: tuple[float, float, float],
    body_local_quat_xyzw: tuple[float, float, float, float],
) -> None:
    """Randomize a base-object lever root and reset its nested handle to the matching rest pose."""
    if env_ids is None:
        return

    object_pose_tensor = _sample_root_pose_xy_yaw(
        env=env,
        env_ids=env_ids,
        base_position_xyz=base_position_xyz,
        base_yaw_rad=base_yaw_rad,
        xy_jitter=xy_jitter,
        z_jitter=z_jitter,
        yaw_jitter_rad=yaw_jitter_rad,
    )
    asset = env.scene[object_name]
    asset.set_world_poses(
        positions=object_pose_tensor[:, :3],
        orientations=object_pose_tensor[:, [6, 3, 4, 5]],
        indices=env_ids,
    )
    _reset_internal_rigid_body_to_pose(
        env=env,
        env_ids=env_ids,
        object_name=object_name,
        body_suffix=body_suffix,
        object_pose_tensor=object_pose_tensor,
        object_scale=object_scale,
        body_local_pos=body_local_pos,
        body_local_quat_xyzw=body_local_quat_xyzw,
    )


def randomize_background_visibility(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    background_names: list[str],
) -> None:
    """Pick one preloaded background per reset and hide the others."""
    if env_ids is None:
        return
    assert (
        background_names
    ), "randomize_background_visibility requires at least one background."

    choices = torch.randint(
        low=0, high=len(background_names), size=(len(env_ids),), device=env.device
    )
    for background_index, background_name in enumerate(background_names):
        visible_env_ids = env_ids[choices == background_index]
        hidden_env_ids = env_ids[choices != background_index]
        if len(visible_env_ids) > 0:
            _set_scene_visibility(env.scene[background_name], True, visible_env_ids)
        if len(hidden_env_ids) > 0:
            _set_scene_visibility(env.scene[background_name], False, hidden_env_ids)


def _set_scene_visibility(asset, visible: bool, env_ids: torch.Tensor) -> None:
    """Set visibility on Isaac Lab assets and base XformPrimView USD roots."""
    try:
        asset.set_visibility(visible, env_ids=env_ids)
    except TypeError as exc:
        if "env_ids" not in str(exc):
            raise
        prims = getattr(asset, "_prims", [])
        num_indices = len(prims)
        if num_indices == 0:
            return
        visibility = torch.full((num_indices,), visible, dtype=torch.bool)
        asset.set_visibility(visibility)
