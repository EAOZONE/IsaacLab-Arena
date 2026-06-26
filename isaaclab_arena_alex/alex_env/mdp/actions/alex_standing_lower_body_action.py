# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
    ALEX_WBC_VERSION_STANDING_PD,
)
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.wbc_policy_factory import get_alex_wbc_policy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena_alex.alex_env.mdp.actions.alex_standing_lower_body_action_cfg import (
        AlexStandingLowerBodyActionCfg,
    )


class AlexStandingLowerBodyAction(ActionTerm):
    """Closed-loop lower-body controller that keeps Alex upright while the arms teleoperate.

    This action term consumes zero entries from the environment action vector. It runs in
    parallel with the Pink IK upper-body action term and writes position targets for the
    leg and spine joints each step.
    """

    cfg: AlexStandingLowerBodyActionCfg

    _asset: Articulation

    def __init__(self, cfg: AlexStandingLowerBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        assert self._joint_names == list(self.cfg.joint_names), (
            f"Lower-body joint resolution mismatch: {self._joint_names} vs {self.cfg.joint_names}"
        )
        self._num_joints = len(self._joint_ids)
        self._joint_ids_tensor = torch.tensor(self._joint_ids, device=self.device, dtype=torch.long)

        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

        self.wbc_policy = get_alex_wbc_policy(
            self.cfg.wbc_version,
            num_envs=self.num_envs,
            joint_names=tuple(self.cfg.joint_names),
            model_path=self.cfg.model_path,
            device=str(self.device),
        )

        self._pelvis_body_idx = self._asset.body_names.index(self.cfg.pelvis_link_name)
        default_joint_pos = wp.to_torch(self._asset.data.default_joint_pos)[0, self._joint_ids_tensor].cpu().numpy()
        if hasattr(self.wbc_policy, "set_default_lower_body_positions"):
            self.wbc_policy.set_default_lower_body_positions(default_joint_pos)
        self._last_rl_action = np.zeros((self.num_envs, self._num_joints), dtype=np.float32)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def get_wbc_policy(self):
        return self.wbc_policy

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions
        joint_pos = wp.to_torch(self._asset.data.joint_pos).index_select(1, self._joint_ids_tensor)
        joint_vel = wp.to_torch(self._asset.data.joint_vel).index_select(1, self._joint_ids_tensor)
        default_joint_pos = wp.to_torch(self._asset.data.default_joint_pos).index_select(1, self._joint_ids_tensor)
        pelvis_quat_wxyz = wp.to_torch(self._asset.data.body_quat_w)[:, self._pelvis_body_idx]
        base_ang_vel = wp.to_torch(self._asset.data.root_ang_vel_b)
        projected_gravity = wp.to_torch(self._asset.data.projected_gravity_b)

        self.wbc_policy.set_observation(
            {
                "joint_pos": (joint_pos - default_joint_pos).detach().cpu().numpy(),
                "joint_vel": joint_vel.detach().cpu().numpy(),
                "pelvis_quat_wxyz": pelvis_quat_wxyz.detach().cpu().numpy(),
                "base_ang_vel": base_ang_vel.detach().cpu().numpy(),
                "projected_gravity": projected_gravity.detach().cpu().numpy(),
                "last_action": self._last_rl_action,
            }
        )
        action = self.wbc_policy.get_action()
        targets = torch.as_tensor(action["joint_targets"], device=self.device, dtype=torch.float32)
        self._processed_actions[:] = targets
        if "raw_action" in action:
            self._last_rl_action = np.asarray(action["raw_action"], dtype=np.float32)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.wbc_policy.reset(env_ids)
        self._last_rl_action[env_ids] = 0.0
