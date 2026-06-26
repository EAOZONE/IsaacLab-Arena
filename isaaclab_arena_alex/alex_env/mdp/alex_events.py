# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_standing_lower_body_policy(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Reset the Alex lower-body standing controller on environment reset."""
    env.action_manager.get_term("lower_body_standing").wbc_policy.reset(env_ids)
