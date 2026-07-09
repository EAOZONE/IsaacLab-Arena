# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pink IK action term with a bounded delta-pose action space, safe for RL.

``PinkInverseKinematicsAction`` (and its Arena subclass
:class:`~isaaclab_arena.embodiments.common.pink_ik_failure_tracking.IKFailureTrackingPinkInverseKinematicsAction`)
treats each wrist's raw ``[pos(3), quat(4)]`` action slot as an *absolute* env-local target, with
no offset or scale (``PinkInverseKinematicsActionCfg`` has no ``scale``/``offset`` fields). A
freshly-initialized RL policy's mean output is ~0, and an all-zero quaternion is a degenerate,
zero-norm rotation: ``matrix_from_quat`` divides by its squared norm, producing NaN and crashing
the IK solve on nearly every step. This action term fixes that by reinterpreting the same 7 raw
floats per wrist as a *bounded delta* from the wrist's current live pose, so a zero action is
always a valid no-op (hold current pose) instead of a degenerate target.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import normalize, quat_mul

from isaaclab_arena.embodiments.common.pink_ik_failure_tracking import IKFailureTrackingPinkInverseKinematicsAction

_IDENTITY_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)


class RLPinkIKAction(IKFailureTrackingPinkInverseKinematicsAction):
    """Pink IK action term whose raw per-wrist ``[pos(3), quat(4)]`` slot is a bounded delta.

    ``target_pos = current_pos + tanh(raw_pos) * cfg.position_scale`` is bounded to
    ``cfg.position_scale`` meters regardless of the raw policy output's magnitude.

    ``target_quat = quat_mul(normalize(raw_quat + identity), current_quat)`` -- adding the
    identity quaternion (``(0, 0, 0, 1)`` in this repo's x, y, z, w convention) before
    normalizing guarantees a non-degenerate unit quaternion for any input: ``raw_quat == 0``
    normalizes to exactly identity (a perfect hold), and the only zero-norm case is the
    single point ``raw_quat == (0, 0, 0, -2)``, not a systematic failure mode like the
    unscaled absolute interpretation.

    Wrist link order matches ``cfg.target_eef_link_names``'s insertion order (``"left"`` then
    ``"right"`` for Alex), the same convention already assumed by
    ``isaaclab_arena/scripts/imitation_learning/record_scripted_lever_demos.py``'s
    ``_pink_ik_action`` when assembling teleop/demo actions in the same raw layout.
    """

    cfg: RLPinkIKActionCfg

    def __init__(self, cfg: RLPinkIKActionCfg, env):
        super().__init__(cfg, env)
        wrist_link_names = list(cfg.target_eef_link_names.values())
        body_ids, _ = self._asset.find_bodies(wrist_link_names)
        self._wrist_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        self._wrist_dim = len(wrist_link_names) * self.pose_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        wrist_actions = actions[:, : self._wrist_dim]
        rest_actions = actions[:, self._wrist_dim :]

        body_state = wp.to_torch(self._asset.data.body_link_state_w)
        current_pos = body_state[:, self._wrist_body_ids, :3] - self._env.scene.env_origins.unsqueeze(1)
        current_quat = body_state[:, self._wrist_body_ids, 3:7]

        identity_quat = torch.tensor(_IDENTITY_QUAT_XYZW, dtype=actions.dtype, device=self.device)
        target_slots = []
        for wrist_idx in range(self._wrist_body_ids.shape[0]):
            pos_start = wrist_idx * self.pose_dim
            raw_pos = wrist_actions[:, pos_start : pos_start + self.position_dim]
            raw_quat = wrist_actions[:, pos_start + self.position_dim : pos_start + self.pose_dim]

            target_pos = current_pos[:, wrist_idx] + torch.tanh(raw_pos) * self.cfg.position_scale
            delta_quat = normalize(raw_quat + identity_quat)
            target_quat = quat_mul(delta_quat, current_quat[:, wrist_idx])

            target_slots.append(target_pos)
            target_slots.append(target_quat)

        transformed_actions = torch.cat(target_slots + [rest_actions], dim=1)
        super().process_actions(transformed_actions)


@configclass
class RLPinkIKActionCfg(PinkInverseKinematicsActionCfg):
    """Configuration for :class:`RLPinkIKAction`."""

    class_type: type = RLPinkIKAction

    position_scale: float = 0.1
    """Max per-step position delta (m), applied as ``tanh(raw_action) * position_scale``."""
