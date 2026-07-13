# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for manager terms evaluated outside an Isaac Lab manager."""

import torch

from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg, TerminationTermCfg


def _resolve_scene_entities(value, env) -> None:
    if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    elif isinstance(value, ManagerTermBaseCfg):
        for nested_value in value.params.values():
            _resolve_scene_entities(nested_value, env)
    elif isinstance(value, dict):
        for nested_value in value.values():
            _resolve_scene_entities(nested_value, env)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            _resolve_scene_entities(nested_value, env)


class _ExtractedManagerTerm:
    """Restore reset behavior normally supplied by an Isaac Lab manager."""

    def __init__(self, term: ManagerTermBase, env):
        self._term = term
        self._last_episode_step = torch.full_like(env.episode_length_buf, -1)

    def __call__(self, env, **params):
        episode_step = env.episode_length_buf
        reset_ids = (episode_step < self._last_episode_step).nonzero(as_tuple=True)[0]
        if len(reset_ids) > 0:
            self._term.reset(reset_ids)
        self._last_episode_step = episode_step.clone()
        return self._term(env, **params)


def bind_extracted_manager_term(term_cfg: TerminationTermCfg | None, env) -> TerminationTermCfg | None:
    """Instantiate a class-based term removed from its manager configuration.

    Imitation-learning scripts remove success from ``env_cfg.terminations`` so it
    cannot reset an active rollout, then evaluate it manually. Function terms
    already work in that mode; ``ManagerTermBase`` subclasses need the
    construction normally performed by ``TerminationManager``.
    """
    if term_cfg is None:
        return None
    term = term_cfg.func
    if isinstance(term, type) and issubclass(term, ManagerTermBase):
        for value in term_cfg.params.values():
            _resolve_scene_entities(value, env)
        term_cfg.func = _ExtractedManagerTerm(term(term_cfg, env), env)
    return term_cfg
