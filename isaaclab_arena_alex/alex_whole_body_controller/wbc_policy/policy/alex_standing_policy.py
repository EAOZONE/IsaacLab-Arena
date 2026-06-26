# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Classical PD standing controller for Alex (Phase 0)."""

from __future__ import annotations

import numpy as np

from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
    ALEX_STANDING_NOMINAL_JOINT_POS,
)
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.base import AlexWBCPolicy


def gravity_orientation_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project world gravity into the pelvis frame (wxyz quaternion, shape (N, 4))."""
    assert quat_wxyz.ndim == 2 and quat_wxyz.shape[1] == 4
    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    return np.stack(
        [
            gravity[0] * (w * w + x * x - y * y - z * z)
            + gravity[1] * 2 * (x * y + w * z)
            + gravity[2] * 2 * (x * z - w * y),
            gravity[0] * 2 * (x * y - w * z)
            + gravity[1] * (w * w - x * x + y * y - z * z)
            + gravity[2] * 2 * (y * z + w * x),
            gravity[0] * 2 * (x * z + w * y)
            + gravity[1] * 2 * (y * z - w * x)
            + gravity[2] * (w * w - x * x - y * y + z * z),
        ],
        axis=1,
    )


class AlexStandingPolicy(AlexWBCPolicy):
    """Hold a nominal crouch and reject pelvis tilt using ankle/hip corrections."""

    def __init__(
        self,
        *,
        num_envs: int,
        joint_names: tuple[str, ...] = ALEX_LOWER_BODY_JOINT_NAMES,
        nominal_joint_pos: dict[str, float] | None = None,
        kp_roll: float = 0.35,
        kp_pitch: float = 0.45,
        kd_joint_vel: float = 0.02,
    ):
        self.num_envs = num_envs
        self.joint_names = joint_names
        self.nominal_joint_pos = dict(nominal_joint_pos or ALEX_STANDING_NOMINAL_JOINT_POS)
        self.kp_roll = kp_roll
        self.kp_pitch = kp_pitch
        self.kd_joint_vel = kd_joint_vel

        self._nominal = np.zeros((num_envs, len(joint_names)), dtype=np.float32)
        for idx, name in enumerate(joint_names):
            self._nominal[:, idx] = self.nominal_joint_pos.get(name, 0.0)

        self._name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        self.observation: dict[str, object] = {}

    def reset(self, env_ids: object | None = None) -> None:
        self.observation = {}

    def get_action(self) -> dict[str, object]:
        joint_pos = np.asarray(self.observation["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(self.observation["joint_vel"], dtype=np.float32)
        pelvis_quat_wxyz = np.asarray(self.observation["pelvis_quat_wxyz"], dtype=np.float32)

        targets = self._nominal.copy()
        if self.kd_joint_vel > 0.0:
            targets -= self.kd_joint_vel * joint_vel

        gravity_body = gravity_orientation_wxyz(pelvis_quat_wxyz)
        roll_error = gravity_body[:, 0]
        pitch_error = gravity_body[:, 1]

        self._apply_axis_correction(targets, roll_error, axis="roll")
        self._apply_axis_correction(targets, pitch_error, axis="pitch")

        return {"joint_targets": targets}

    def _apply_axis_correction(self, targets: np.ndarray, error: np.ndarray, *, axis: str) -> None:
        if axis == "roll":
            kp = self.kp_roll
            joint_map = {
                "LEFT_HIP_X": 0.35,
                "RIGHT_HIP_X": 0.35,
                "LEFT_ANKLE_X": 0.65,
                "RIGHT_ANKLE_X": 0.65,
            }
        elif axis == "pitch":
            kp = self.kp_pitch
            joint_map = {
                "LEFT_HIP_Y": 0.30,
                "RIGHT_HIP_Y": 0.30,
                "LEFT_KNEE_Y": 0.15,
                "RIGHT_KNEE_Y": 0.15,
                "LEFT_ANKLE_Y": 0.55,
                "RIGHT_ANKLE_Y": 0.55,
                "SPINE_Z": 0.20,
            }
        else:
            raise ValueError(f"Unknown axis: {axis}")

        for joint_name, scale in joint_map.items():
            idx = self._name_to_idx[joint_name]
            targets[:, idx] -= kp * scale * error
