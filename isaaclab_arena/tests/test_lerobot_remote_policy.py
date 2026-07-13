# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab_arena.policy.lerobot_remote_policy import LeRobotRemotePolicy


def test_test_obs_new_action_is_reordered_for_pink_ik() -> None:
    action = torch.arange(46, dtype=torch.float32).reshape(1, 46)
    arena = LeRobotRemotePolicy._to_arena_action(action)
    assert arena.shape == (1, 34)
    assert torch.equal(arena[:, :14], action[:, :14])
    assert sorted(arena[0, 14:].tolist()) == list(range(26, 46))
