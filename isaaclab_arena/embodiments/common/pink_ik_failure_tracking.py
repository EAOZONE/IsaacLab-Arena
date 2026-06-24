# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pink IK action term that latches per-env IK-solver failures across an episode."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction


class IKFailureTrackingPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink IK action term that records whether any IK solve failed during the current episode.

    Each :class:`~isaaclab.controllers.pink_ik.PinkIKController` sets ``solve_failed`` on every
    ``compute()`` call. This term ORs those per-env flags into a buffer that persists until the
    env is reset, so a termination/metric can fail episodes where the arm reached the goal only
    because the IK target was silently dropped (the "cheated past an unsolved target" case).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ik_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Per-env latch: ``True`` if any IK solve has failed since the last reset."""

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
