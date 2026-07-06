# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frame bridge between H2Ozone/lever_eef dataset poses and Arena Pink IK wrist targets.

The lever_eef dataset stores gripper poses in the real robot's world frame using IHMC
hand control frames. Arena's ``alex_v2_ability_hands`` Pink IK action term consumes
``LEFT/RIGHT_GRIPPER_Z_LINK`` targets in the env/world frame.

Every quaternion in this stack is **scalar-last (x, y, z, w)**: the lever_eef dataset
columns (qx, qy, qz, qs), Isaac Lab 3.0 ``body_quat_w`` (``wp.quatf``) and
``matrix_from_quat``, and the 34-dim Pink IK action vector (``l_quat_x`` … ``l_quat_w``
at indices 3-6 / 10-13 in the action YAML). No layout conversion happens anywhere in
this module — poses are split, normalized, and composed as xyzw.

Constants were solved from H2Ozone/lever_fingers FK vs lever_eef pose pairs (see
``playback_lerobot_eef_dataset.py`` for the derivation). The pelvis composition keeps
the calibration valid when Alex spawns away from the teleop sandbox origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab_arena_gr00t.streaming.gr00t_eef_ikstream_bridge import (
    EEF_ACTION_DIM,
    LEFT_WRIST_POSE_SLICE,
    RIGHT_WRIST_POSE_SLICE,
    _base_link_pose_in_env,
)

# Replay target (sim world) = T_pelvis_now * T_PELVIS_CALIB^-1 * A_INV * D * B_INV_<hand>
_A_INV = ((0.043138, -0.482041, 0.028456), (0.0, 0.0, 0.0, 1.0))
_B_INV = {
    "left": (
        (0.013252, 0.007675, -0.002249),
        (-0.161324, -0.324033, -0.890589, 0.275369),
    ),
    "right": (
        (-0.073235, -0.007256, -0.00897),
        (-0.178244, -0.374856, -0.909779, 0.003727),
    ),
}
# Pelvis world pose (pos, quat xyzw) in alex_teleop_sandbox where constants were solved
# (its 0-yaw spawn: identity orientation). Must stay in the raw ``body_quat_w`` xyzw
# convention so it cancels against ``_base_link_pose_in_env`` at the calibration spawn.
_PELVIS_CALIB = ((-0.4, -0.48682, 0.94296), (0.0, 0.0, 0.0, 1.0))

_LEFT_WRIST_KEY = "left_wrist_pose"
_RIGHT_WRIST_KEY = "right_wrist_pose"
_LEVER_EEF_NECK_JOINT_NAMES = ("NECK_Z", "NECK_Y")

# robot id -> int32 tensor of neck joint indices on that robot's device
_NECK_JOINT_IDS_CACHE: dict[int, torch.Tensor] = {}

# robot id -> hand-slot permutation (or None when identity)
_HAND_SLOT_PERMUTATION_CACHE: dict[int, list[int] | None] = {}

# First hand slot in the 34-dim Pink IK action vector (after two 7-dim wrist poses).
_HAND_BLOCK_START = 14


def uses_lever_eef_frame_bridge(modality_config_path: str | None) -> bool:
    """Return True when the closed-loop policy should bridge lever_eef dataset frames."""
    return modality_config_path is not None and "lever_eef" in str(modality_config_path)


def split_pose7_xyzw(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 7-vec into ``(pos, quat_xyzw)``, normalizing the quaternion.

    Dataset/policy poses (qx, qy, qz, qs), Isaac ``body_quat_w``, and the Pink IK
    action layout all carry scalar-last quats, so no layout conversion is done.
    """
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    pos = pose7[:3]
    quat = pose7[3:7]
    quat = quat / np.linalg.norm(quat)
    return pos, quat


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    qc = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
    return _quat_mul_xyzw(_quat_mul_xyzw(q, qv), qc)[:3]


def _pose_mul(a: tuple, b: tuple) -> tuple:
    pa, qa = np.asarray(a[0], dtype=np.float64), np.asarray(a[1], dtype=np.float64)
    pb, qb = np.asarray(b[0], dtype=np.float64), np.asarray(b[1], dtype=np.float64)
    return pa + _quat_rotate_xyzw(qa, pb), _quat_mul_xyzw(qa, qb)


def _pose_inv(a: tuple) -> tuple:
    p, q = np.asarray(a[0], dtype=np.float64), np.asarray(a[1], dtype=np.float64)
    qc = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
    return -_quat_rotate_xyzw(qc, p), qc


@dataclass
class LeverEefFrameCalibration:
    """Maps wrist poses between lever_eef dataset coordinates and Arena sim world."""

    world_from_dataset: tuple

    def dataset_pose_to_pink_ik_pose(self, dataset_pose7: np.ndarray, hand: str) -> np.ndarray:
        """Dataset / policy wrist pose -> sim Pink IK target (pos + quat xyzw)."""
        pos, quat_xyzw = split_pose7_xyzw(dataset_pose7)
        pose = _pose_mul(_pose_mul(self.world_from_dataset, (pos, quat_xyzw)), _B_INV[hand])
        quat_xyzw = pose[1] / np.linalg.norm(pose[1])
        return np.concatenate([pose[0], quat_xyzw]).astype(np.float32)

    def sim_pose_to_dataset_pose(self, sim_pose7: np.ndarray, hand: str) -> np.ndarray:
        """Sim GRIPPER_Z_LINK pose -> lever_eef dataset state (pos + quat xyzw)."""
        pos, quat_xyzw = split_pose7_xyzw(sim_pose7)
        dataset_pose = _pose_mul(
            _pose_mul(_pose_inv(self.world_from_dataset), (pos, quat_xyzw)),
            _pose_inv(_B_INV[hand]),
        )
        quat_xyzw = dataset_pose[1] / np.linalg.norm(dataset_pose[1])
        return np.concatenate([dataset_pose[0], quat_xyzw]).astype(np.float32)


def build_lever_eef_calibration(env, env_index: int = 0) -> LeverEefFrameCalibration | None:
    """Build a live calibration from the robot pelvis pose in ``env``."""
    base_pose = _base_link_pose_in_env(env, env_index)
    if base_pose is None:
        return None
    base_pos, base_quat_xyzw = base_pose
    pelvis_now = (base_pos, base_quat_xyzw)
    world_from_dataset = _pose_mul(_pose_mul(pelvis_now, _pose_inv(_PELVIS_CALIB)), _A_INV)
    return LeverEefFrameCalibration(world_from_dataset=world_from_dataset)


def convert_sim_eef_state_to_dataset(
    eef_pose_policy: dict[str, np.ndarray],
    env,
) -> dict[str, np.ndarray]:
    """Convert sim wrist observations into lever_eef dataset-frame policy state."""
    converted = dict(eef_pose_policy)
    if not converted:
        return converted
    num_envs = next(iter(converted.values())).shape[0]
    for env_index in range(num_envs):
        calibration = build_lever_eef_calibration(env, env_index)
        assert calibration is not None, "Could not resolve PELVIS_LINK for lever_eef frame bridge"
        for hand, key in (("left", _LEFT_WRIST_KEY), ("right", _RIGHT_WRIST_KEY)):
            if key not in converted:
                continue
            pose = np.asarray(converted[key][env_index], dtype=np.float32)
            converted[key][env_index] = calibration.sim_pose_to_dataset_pose(pose, hand)
    return converted


def convert_policy_wrist_actions_to_sim(
    robot_action_policy: dict[str, np.ndarray],
    env,
) -> dict[str, np.ndarray]:
    """Convert GR00T wrist action groups from lever_eef dataset frame to Pink IK frame."""
    converted = dict(robot_action_policy)
    wrist_keys = (_LEFT_WRIST_KEY, _RIGHT_WRIST_KEY)
    if not all(key in converted for key in wrist_keys):
        return converted

    sample = converted[_LEFT_WRIST_KEY]
    num_envs = int(sample.shape[0])
    horizon = int(sample.shape[1]) if sample.ndim == 3 else 1

    for hand, key in (("left", _LEFT_WRIST_KEY), ("right", _RIGHT_WRIST_KEY)):
        wrist = np.asarray(converted[key], dtype=np.float32)
        was_2d = wrist.ndim == 2
        if was_2d:
            wrist = wrist[:, None, :]
        for env_index in range(num_envs):
            calibration = build_lever_eef_calibration(env, env_index)
            assert calibration is not None, "Could not resolve PELVIS_LINK for lever_eef frame bridge"
            for step in range(horizon):
                wrist[env_index, step] = calibration.dataset_pose_to_pink_ik_pose(
                    wrist[env_index, step], hand
                )
        converted[key] = wrist[:, 0, :] if was_2d else wrist
    return converted


def convert_dataset_eef_action_to_sim(
    action_np: np.ndarray,
    env,
) -> np.ndarray:
    """Convert a batched 34-dim EEF action chunk from dataset frame to Pink IK frame."""
    action_np = np.asarray(action_np, dtype=np.float32).copy()
    assert action_np.shape[-1] == EEF_ACTION_DIM, (
        f"expected {EEF_ACTION_DIM}-dim EEF action, got shape {action_np.shape}"
    )
    num_envs = action_np.shape[0]
    horizon = action_np.shape[1] if action_np.ndim == 3 else 1
    if action_np.ndim == 2:
        action_np = action_np[:, None, :]

    for env_index in range(num_envs):
        calibration = build_lever_eef_calibration(env, env_index)
        assert calibration is not None, "Could not resolve PELVIS_LINK for lever_eef frame bridge"
        for step in range(horizon):
            left = action_np[env_index, step, LEFT_WRIST_POSE_SLICE]
            right = action_np[env_index, step, RIGHT_WRIST_POSE_SLICE]
            action_np[env_index, step, LEFT_WRIST_POSE_SLICE] = calibration.dataset_pose_to_pink_ik_pose(
                left, "left"
            )
            action_np[env_index, step, RIGHT_WRIST_POSE_SLICE] = calibration.dataset_pose_to_pink_ik_pose(
                right, "right"
            )
    if horizon == 1 and action_np.ndim == 3:
        return action_np[:, 0, :]
    return action_np


def _resolve_hand_slot_permutation(robot) -> list[int] | None:
    """Permutation from teleop-order hand slots to the order Pink IK applies them.

    ``PinkInverseKinematicsAction`` resolves ``hand_joint_names`` with a
    non-order-preserving ``find_joints``, so hand slot ``k`` of the action drives the
    ``k``-th hand joint in *asset* order, not the ``k``-th name in
    ``ABILITY_HAND_TELEOP_JOINT_ORDER``. Returns ``perm`` such that
    ``permuted[k] = teleop_block[perm[k]]`` gives every joint its intended value,
    or ``None`` when the orders already match.
    """
    cache_key = id(robot)
    if cache_key in _HAND_SLOT_PERMUTATION_CACHE:
        return _HAND_SLOT_PERMUTATION_CACHE[cache_key]

    from isaaclab_arena.embodiments.alex.alex import ABILITY_HAND_TELEOP_JOINT_ORDER

    _, applied_order = robot.find_joints(list(ABILITY_HAND_TELEOP_JOINT_ORDER))
    assert sorted(applied_order) == sorted(ABILITY_HAND_TELEOP_JOINT_ORDER), (
        f"unexpected hand joints resolved: {applied_order}"
    )
    teleop_index = {name: i for i, name in enumerate(ABILITY_HAND_TELEOP_JOINT_ORDER)}
    perm = [teleop_index[name] for name in applied_order]
    if perm == list(range(len(perm))):
        perm = None
    _HAND_SLOT_PERMUTATION_CACHE[cache_key] = perm
    return perm


def reorder_hand_targets_for_pink_ik(action: torch.Tensor, env) -> torch.Tensor:
    """Permute the 20 hand slots of a ``(..., 34)`` action so each finger gets its value.

    Only needed for actions built from *semantic* joint names (real-robot lever_eef
    policies); sim-teleop recorded actions already carry the applied convention.
    """
    assert action.shape[-1] == EEF_ACTION_DIM, (
        f"expected {EEF_ACTION_DIM}-dim EEF action, got shape {tuple(action.shape)}"
    )
    unwrapped = getattr(env, "unwrapped", env)
    perm = _resolve_hand_slot_permutation(unwrapped.scene["robot"])
    if perm is None:
        return action
    hand_slots = torch.tensor(
        [_HAND_BLOCK_START + p for p in perm], dtype=torch.long, device=action.device
    )
    action = action.clone()
    action[..., _HAND_BLOCK_START:] = action[..., hand_slots]
    return action


def _resolve_neck_joint_ids(robot) -> torch.Tensor:
    cache_key = id(robot)
    cached = _NECK_JOINT_IDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    neck_ids_list, resolved = robot.find_joints(
        list(_LEVER_EEF_NECK_JOINT_NAMES), preserve_order=True
    )
    assert list(resolved) == list(_LEVER_EEF_NECK_JOINT_NAMES), (
        f"expected neck joints {_LEVER_EEF_NECK_JOINT_NAMES}, got {list(resolved)}"
    )
    neck_ids = torch.tensor(neck_ids_list, dtype=torch.int32, device=robot.device)
    _NECK_JOINT_IDS_CACHE[cache_key] = neck_ids
    return neck_ids


def write_lever_eef_neck_targets(
    env,
    neck_targets: np.ndarray | torch.Tensor,
    env_mask: torch.Tensor | None = None,
) -> None:
    """Write neck joint targets kinematically (not part of the 34-dim Pink IK action).

    Args:
        env: Gym env (wrapped or unwrapped) with ``scene["robot"]``.
        neck_targets: ``(num_envs, 2)`` or ``(2,)`` neck joint positions in sim order
            (``NECK_Z``, ``NECK_Y``).
        env_mask: Optional bool ``(num_envs,)`` — when set, True marks envs that should
            be skipped (e.g. SyncedBatchActionScheduler hold state).
    """
    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene["robot"]
    device = robot.device
    neck = torch.as_tensor(neck_targets, dtype=torch.float32, device=device)
    if neck.ndim == 1:
        neck = neck.unsqueeze(0)

    neck_ids = _resolve_neck_joint_ids(robot)
    if env_mask is not None:
        active = (~env_mask).nonzero(as_tuple=False).flatten()
    else:
        active = torch.arange(neck.shape[0], device=device)

    for env_idx in active.tolist():
        row = neck[env_idx].unsqueeze(0)
        env_ids = torch.tensor([env_idx], dtype=torch.int64, device=device)
        robot.write_joint_position_to_sim_index(
            position=row, joint_ids=neck_ids, env_ids=env_ids
        )
        robot.set_joint_position_target_index(
            target=row, joint_ids=neck_ids, env_ids=env_ids
        )
