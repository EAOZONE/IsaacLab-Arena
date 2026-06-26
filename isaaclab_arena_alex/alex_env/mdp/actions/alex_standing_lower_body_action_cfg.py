# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab_arena_alex.alex_env.mdp.actions.alex_standing_lower_body_action import AlexStandingLowerBodyAction
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
    ALEX_WBC_VERSION_STANDING_PD,
)


@configclass
class AlexStandingLowerBodyActionCfg(ActionTermCfg):
    """Configuration for the Alex classical / RL lower-body standing controller."""

    class_type: type[ActionTerm] = AlexStandingLowerBodyAction

    preserve_order: bool = True
    joint_names: list[str] = list(ALEX_LOWER_BODY_JOINT_NAMES)
    pelvis_link_name: str = "PELVIS_LINK"
    wbc_version: str = ALEX_WBC_VERSION_STANDING_PD
    model_path: str | None = None
