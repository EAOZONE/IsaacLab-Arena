# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pink IK action term that latches per-env IK-solver failures across an episode."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction


class IKFailureTrackingPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink IK action term that records whether any IK solve failed during the current episode.

    Each :class:`~isaaclab.controllers.pink_ik.PinkIKController` sets ``solve_failed`` on every
    ``compute()`` call. This term ORs those per-env flags into a buffer that persists until the
    env is reset, so a termination/metric can fail episodes where the arm reached the goal only
    because the IK target was silently dropped (the "cheated past an unsolved target" case).

    This also fixes up a hand-joint ordering mismatch in the base action term:
    ``PinkInverseKinematicsAction._initialize_joint_info`` resolves ``cfg.hand_joint_names`` via
    ``Articulation.find_joints(...)`` *without* ``preserve_order=True``, so the actual
    ``self._hand_joint_ids`` end up sorted by the joints' ascending indices in the URDF (e.g.
    Alex's ability hands declare each hand as index/middle/ring/pinky/**thumb**, so thumb lands
    right after pinky instead of at the end) rather than in ``cfg.hand_joint_names``'s given
    order. Every caller that builds an action tensor (teleop, recorded datasets, GR00T inference)
    assumes the hand-joint block matches ``cfg.hand_joint_names``'s order, so left uncorrected
    this silently sends each hand-joint target to the wrong physical joint. ``process_actions``
    below permutes the incoming hand-joint block back into ``self._hand_joint_names``'s resolved
    order right before handing off to the base implementation, so callers can keep assuming
    ``cfg.hand_joint_names``'s order without the base class's resolved order leaking through.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ik_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Per-env latch: ``True`` if any IK solve has failed since the last reset."""

        # self._hand_joint_names (set by the base class) is self.cfg.hand_joint_names resolved via
        # find_joints without preserve_order=True, so it may differ in order (see class docstring).
        # _hand_joint_permutation[j] is the cfg.hand_joint_names index whose value belongs at
        # resolved-order slot j, i.e. `hand_actions[:, self._hand_joint_names[j]'s cfg index]`.
        self._hand_joint_permutation = torch.tensor(
            [self.cfg.hand_joint_names.index(name) for name in self._hand_joint_names],
            dtype=torch.long,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        hand_dim = self.hand_joint_dim
        wrist_actions = actions[:, :-hand_dim]
        hand_actions = actions[:, -hand_dim:]
        reordered_hand_actions = hand_actions[:, self._hand_joint_permutation]
        super().process_actions(torch.cat([wrist_actions, reordered_hand_actions], dim=1))

    def _compute_ik_solutions(self) -> torch.Tensor:
        solutions = super()._compute_ik_solutions()
        failed = torch.tensor(
            [bool(getattr(c, "solve_failed", False)) for c in self._ik_controllers],
            device=self.device,
            dtype=torch.bool,
        )
        self.ik_failed |= failed
        return solutions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self.ik_failed[:] = False
        else:
            self.ik_failed[env_ids] = False
