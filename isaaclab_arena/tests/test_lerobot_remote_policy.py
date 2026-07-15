# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from types import SimpleNamespace

pytest.importorskip("isaaclab")

from isaaclab_arena.policy.lerobot_remote_policy import (
    LeRobotRemotePolicy,
    _quat_wxyz_to_xyzw,
)


def test_test_obs_new_action_is_reordered_for_pink_ik() -> None:
    action = torch.arange(46, dtype=torch.float32).reshape(1, 46)
    arena = LeRobotRemotePolicy._to_arena_action(action)
    assert arena.shape == (1, 34)
    assert torch.equal(arena[:, :14], action[:, :14])
    assert sorted(arena[0, 14:].tolist()) == list(range(26, 46))


def test_body_quat_is_converted_to_xyzw() -> None:
    wxyz = torch.tensor([[0.5, 0.1, 0.2, 0.3]])
    assert torch.allclose(
        _quat_wxyz_to_xyzw(wxyz), wxyz
    )


def test_test_obs_new_recorders_pack_state_and_action() -> None:
    from isaaclab_arena.embodiments.alex.alex import ABILITY_HAND_TELEOP_JOINT_ORDER
    from isaaclab_arena.utils.isaaclab_utils.recorders import (
        _TEST_OBS_NEW_GROUPED_FROM_PINK,
        test_obs_new_action,
        test_obs_new_state,
    )

    body_names = [
        "LEFT_GRIPPER_Z_LINK",
        "RIGHT_GRIPPER_Z_LINK",
        "LEFT_WRIST_Z_LINK",
        "RIGHT_WRIST_Z_LINK",
        "HEAD_LINK",
    ]
    joint_names = list(ABILITY_HAND_TELEOP_JOINT_ORDER) + ["SPINE_Z", "SPINE_Y"]

    class _Robot:
        data = SimpleNamespace(
            body_names=body_names,
            body_pos_w=torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
            body_quat_w=torch.tensor(
                [
                    [
                        [1.0, 0.1, 0.2, 0.3],
                        [1.0, 0.4, 0.5, 0.6],
                        [1.0, 0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2, 1.3],
                        [1.0, 1.4, 1.5, 1.6],
                    ]
                ]
            ),
            joint_names=joint_names,
            joint_pos=torch.arange(len(joint_names), dtype=torch.float32).reshape(
                1, -1
            ),
        )

        def find_bodies(self, names):
            return [body_names.index(names[0])], None

        def find_joints(self, names, preserve_order=False):
            return [joint_names.index(name) for name in names], None

    class _Scene(dict):
        pass

    scene = _Scene(robot=_Robot())
    scene.env_origins = torch.zeros((1, 3), dtype=torch.float32)
    env = SimpleNamespace(
        scene=scene,
        action_manager=SimpleNamespace(
            action=torch.arange(34, dtype=torch.float32).reshape(1, 34)
        ),
    )

    state = test_obs_new_state(env)
    action = test_obs_new_action(env)

    assert state.shape == (1, 48)
    assert action.shape == (1, 46)
    assert torch.equal(action[:, :14], env.action_manager.action[:, :14])
    assert torch.equal(
        action[:, 26:46],
        env.action_manager.action[:, 14:34][:, _TEST_OBS_NEW_GROUPED_FROM_PINK],
    )
