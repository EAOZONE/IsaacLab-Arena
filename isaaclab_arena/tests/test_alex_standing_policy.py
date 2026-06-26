# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Alex classical standing controller (no simulation)."""

import numpy as np

from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
)
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_standing_policy import (
    AlexStandingPolicy,
    gravity_orientation_wxyz,
)


def test_gravity_orientation_is_down_when_upright():
  quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
  gravity = gravity_orientation_wxyz(quat)[0]
  np.testing.assert_allclose(gravity, [0.0, 0.0, -1.0], atol=1e-5)


def test_standing_policy_corrects_forward_tilt():
  policy = AlexStandingPolicy(num_envs=1, kd_joint_vel=0.0)
  nominal = {name: 0.0 for name in ALEX_LOWER_BODY_JOINT_NAMES}
  policy.nominal_joint_pos = nominal
  policy._nominal[:] = 0.0

  # Small pitch forward: gravity y-component becomes positive in the pelvis frame.
  policy.set_observation(
      {
          "joint_pos": np.zeros((1, len(ALEX_LOWER_BODY_JOINT_NAMES)), dtype=np.float32),
          "joint_vel": np.zeros((1, len(ALEX_LOWER_BODY_JOINT_NAMES)), dtype=np.float32),
          "pelvis_quat_wxyz": np.array([[0.9962, 0.0872, 0.0, 0.0]], dtype=np.float32),
      }
  )
  targets = policy.get_action()["joint_targets"][0]
  name_to_idx = {name: idx for idx, name in enumerate(ALEX_LOWER_BODY_JOINT_NAMES)}
  assert targets[name_to_idx["LEFT_ANKLE_Y"]] > 0.0
  assert targets[name_to_idx["RIGHT_ANKLE_Y"]] > 0.0


def test_standing_policy_joint_velocity_damping():
  policy = AlexStandingPolicy(num_envs=1, kd_joint_vel=0.1)
  policy.set_observation(
      {
          "joint_pos": np.zeros((1, len(ALEX_LOWER_BODY_JOINT_NAMES)), dtype=np.float32),
          "joint_vel": np.ones((1, len(ALEX_LOWER_BODY_JOINT_NAMES)), dtype=np.float32),
          "pelvis_quat_wxyz": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
      }
  )
  targets = policy.get_action()["joint_targets"]
  assert np.all(targets < policy._nominal)
