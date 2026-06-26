# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
    ALEX_WBC_VERSION_RL,
    ALEX_WBC_VERSION_STANDING_PD,
)
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_rl_standing_policy import (
    AlexRLStandingPolicy,
)
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_standing_policy import AlexStandingPolicy
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.base import AlexWBCPolicy


def get_alex_wbc_policy(
    wbc_version: str,
    *,
    num_envs: int,
    joint_names: tuple[str, ...] = ALEX_LOWER_BODY_JOINT_NAMES,
    model_path: str | None = None,
    device: str = "cpu",
) -> AlexWBCPolicy:
    """Instantiate the Alex lower-body policy for the requested backend."""
    if wbc_version == ALEX_WBC_VERSION_STANDING_PD:
        return AlexStandingPolicy(num_envs=num_envs, joint_names=joint_names)
    if wbc_version == ALEX_WBC_VERSION_RL:
        return AlexRLStandingPolicy(
            num_envs=num_envs,
            joint_names=joint_names,
            model_path=model_path,
            device=device,
        )
    raise ValueError(
        f"Invalid Alex WBC version: {wbc_version}. Supported: {ALEX_WBC_VERSION_STANDING_PD}, {ALEX_WBC_VERSION_RL}"
    )
