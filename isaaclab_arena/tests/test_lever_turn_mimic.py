# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for lever-turn Mimic configuration and stateful signals."""

import torch
from types import SimpleNamespace


def test_lever_mimic_config_uses_active_and_stationary_arms():
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.tasks.common.lever_turn_mimic import LeverTurnMimicEnvCfg

    cfg = LeverTurnMimicEnvCfg(arm_mode=ArmMode.RIGHT, lever_object_name="test_lever")

    assert set(cfg.subtask_configs) == {"left", "right"}
    assert len(cfg.subtask_configs["right"]) == 2
    assert cfg.subtask_configs["right"][0].object_ref == "test_lever"
    assert cfg.subtask_configs["right"][0].subtask_term_signal == "lever_engaged"
    assert cfg.subtask_configs["right"][1].subtask_term_signal is None
    assert len(cfg.subtask_configs["left"]) == 1
    assert cfg.subtask_configs["left"][0].action_noise == 0.0


def _make_stateful_term(term_type):
    term = object.__new__(term_type)
    term._tracking = torch.zeros(1, dtype=torch.bool)
    term._rest_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    if term_type.__name__ == "LeverTurnSuccess":
        term._steps_above_threshold = torch.zeros(1, dtype=torch.long)
        term._last_processed_step = torch.full((1,), -1, dtype=torch.long)
    else:
        term._engaged = torch.zeros(1, dtype=torch.bool)
        term._last_episode_step = torch.full((1,), -1, dtype=torch.long)
    return term


def test_lever_success_counts_each_sim_step_once_and_resets(monkeypatch):
    from isaaclab_arena.tasks.rewards.lever_turn_rewards import HingeAngleFromRest, LeverTurnSuccess

    env = SimpleNamespace(episode_length_buf=torch.tensor([1]), angle=torch.tensor([1.0]))
    monkeypatch.setattr(HingeAngleFromRest, "__call__", lambda self, env, object_cfg: env.angle)
    term = _make_stateful_term(LeverTurnSuccess)
    term._DEBOUNCE_STEPS = 2

    assert not bool(term(env, None, 0.35)[0])
    assert not bool(term(env, None, 0.35)[0]), "duplicate evaluation must not advance the debounce"
    env.episode_length_buf[:] = 2
    assert bool(term(env, None, 0.35)[0])

    env.episode_length_buf[:] = 0
    assert not bool(term(env, None, 0.35)[0])
    assert term._steps_above_threshold.item() == 1


def test_lever_engaged_signal_latches_until_episode_reset(monkeypatch):
    from isaaclab_arena.tasks.rewards.lever_turn_rewards import HingeAngleFromRest, LeverEngaged

    env = SimpleNamespace(episode_length_buf=torch.tensor([1]), angle=torch.tensor([0.0]))
    monkeypatch.setattr(HingeAngleFromRest, "__call__", lambda self, env, object_cfg: env.angle)
    term = _make_stateful_term(LeverEngaged)

    assert not bool(term(env, None, 0.05)[0])
    env.episode_length_buf[:] = 2
    env.angle[:] = 0.1
    assert bool(term(env, None, 0.05)[0])
    env.episode_length_buf[:] = 3
    env.angle[:] = 0.0
    assert bool(term(env, None, 0.05)[0])
    env.episode_length_buf[:] = 0
    assert not bool(term(env, None, 0.05)[0])


def test_extracted_manager_term_resets_on_episode_counter_wrap():
    from isaaclab_arena.utils.isaaclab_utils.manager_terms import _ExtractedManagerTerm

    class FakeTerm:
        def __init__(self):
            self.reset_ids = []

        def reset(self, env_ids):
            self.reset_ids.append(env_ids.tolist())

        def __call__(self, env, **params):
            return params["value"]

    env = SimpleNamespace(episode_length_buf=torch.tensor([5, 5]))
    term = FakeTerm()
    wrapper = _ExtractedManagerTerm(term, env)
    assert wrapper(env, value=1) == 1
    env.episode_length_buf[:] = torch.tensor([0, 6])
    assert wrapper(env, value=2) == 2
    assert term.reset_ids == [[0]]
