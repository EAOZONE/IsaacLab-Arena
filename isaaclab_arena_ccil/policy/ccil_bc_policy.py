# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CCIL behavioral-cloning policy for closed-loop evaluation in Isaac Lab Arena.

The model is trained offline with CCIL (custom d3rlpy fork, Python 3.8) and exported to
a dependency-free artifact. This policy loads that artifact and runs pure PyTorch at
inference — no d3rlpy, no remote server.

Two artifact formats are supported (resolved automatically):

1. **TorchScript** (preferred): a module saved by d3rlpy ``save_policy`` mapping a raw
   observation ``(B, obs_dim)`` to a raw action ``(B, act_dim)`` with the standard /
   min-max scalers baked in. Loaded via ``torch.jit.load``.
2. **state_dict + meta.json** (fallback for TorchScript cross-version issues): a plain MLP
   ``state_dict`` plus ``meta.json`` describing ``hidden_units``/``activation`` and the
   observation (standard) and action (min-max) normalization, reconstructed here.

Observation is ``robot_joint_pos`` (49-dof). The action is the 34-dim raw Pink IK action —
left/right EE target poses (pos 3 + quat 4 each = 14) followed by 20 ability-hand finger
joints — so it must be applied through an IK-in-the-loop embodiment (``alex_ability_hands``)
that resolves the EE targets to a whole-body solution, not the direct joint-position
embodiment. (Train on ``processed_actions`` instead for the legacy direct-joint path.)
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import json
import logging
import os
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_ccil.policy.config.ccil_bc_policy_config import CCILBCPolicyConfig

logger = logging.getLogger(__name__)

_ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
    "elu": torch.nn.ELU,
}


def _build_mlp(input_dim: int, output_dim: int, hidden_units: list[int], activation: str) -> torch.nn.Sequential:
    """Build a plain MLP matching d3rlpy's default vector encoder + action head."""
    act_cls = _ACTIVATIONS[activation.lower()]
    layers: list[torch.nn.Module] = []
    prev = input_dim
    for h in hidden_units:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(torch.nn.Linear(prev, output_dim))
    return torch.nn.Sequential(*layers)


class CCILBCPolicy(PolicyBase):
    """CCIL behavioral-cloning policy (state-based MLP) running in-process in Arena."""

    name = "ccil_bc"
    config_class = CCILBCPolicyConfig

    def __init__(self, config: CCILBCPolicyConfig):
        super().__init__(config)
        self.device = config.policy_device
        self.state_key = config.state_key
        self.num_envs = config.num_envs

        assert os.path.exists(config.model_path), f"CCIL model not found: {config.model_path}"
        self._meta: dict = {}
        if config.meta_path is not None:
            assert os.path.exists(config.meta_path), f"CCIL meta not found: {config.meta_path}"
            with open(config.meta_path) as f:
                self._meta = json.load(f)

        self._jit_module: torch.jit.ScriptModule | None = None
        self._mlp: torch.nn.Sequential | None = None
        self._obs_mean: torch.Tensor | None = None
        self._obs_std: torch.Tensor | None = None
        self._act_min: torch.Tensor | None = None
        self._act_max: torch.Tensor | None = None

        self._load_model(config.model_path)
        self.task_description: str | None = None

    # ---------------------- model loading -------------------

    def _load_model(self, model_path: str) -> None:
        """Load the TorchScript module if possible, else reconstruct the MLP from meta."""
        try:
            self._jit_module = torch.jit.load(model_path, map_location=self.device)
            self._jit_module.eval()
            logger.info(f"CCILBCPolicy: loaded TorchScript policy from {model_path}")
            return
        except (RuntimeError, ValueError) as e:
            logger.warning(f"CCILBCPolicy: torch.jit.load failed ({e}); falling back to state_dict + meta.")

        assert self._meta, "state_dict fallback requires --meta_path pointing at ccil_bc_meta.json"
        input_dim = int(self._meta["input_dim"])
        output_dim = int(self._meta["output_dim"])
        hidden_units = list(self._meta["hidden_units"])
        activation = str(self._meta.get("activation", "relu"))
        self._mlp = _build_mlp(input_dim, output_dim, hidden_units, activation).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        # Accept either a raw state_dict or a checkpoint dict wrapping it.
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self._mlp.load_state_dict(state_dict)
        self._mlp.eval()

        obs_norm = self._meta["obs_norm"]
        self._obs_mean = torch.tensor(obs_norm["mean"], dtype=torch.float, device=self.device)
        self._obs_std = torch.tensor(obs_norm["std"], dtype=torch.float, device=self.device)
        act_norm = self._meta["action_norm"]
        self._act_min = torch.tensor(act_norm["min"], dtype=torch.float, device=self.device)
        self._act_max = torch.tensor(act_norm["max"], dtype=torch.float, device=self.device)
        logger.info(f"CCILBCPolicy: reconstructed MLP {input_dim}->{hidden_units}->{output_dim} from {model_path}")

    def _forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map a raw observation batch to a raw action batch."""
        if self._jit_module is not None:
            return self._jit_module(obs)
        # Plain-MLP fallback: standard-scale obs, run MLP, inverse min-max scale action.
        scaled_obs = (obs - self._obs_mean) / self._obs_std
        y = self._mlp(scaled_obs)  # network output in [-1, 1] (min-max scaled action space)
        return (y + 1.0) / 2.0 * (self._act_max - self._act_min) + self._act_min

    # ---------------------- inference -------------------

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        assert "policy" in observation and self.state_key in observation["policy"], (
            f"observation missing 'policy/{self.state_key}'"
        )
        obs = observation["policy"][self.state_key].to(device=self.device, dtype=torch.float)
        with torch.inference_mode():
            action = self._forward(obs)
        if not torch.isfinite(action).all():
            logger.warning("CCILBCPolicy: non-finite action; substituting zeros for affected entries.")
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        return action.to(device=env.unwrapped.device, dtype=torch.float)

    # ---------------------- CLI helpers -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("CCIL BC Policy", "Arguments for the CCIL behavioral-cloning policy.")
        group.add_argument(
            "--model_path", type=str, required=True, help="CCIL policy artifact (TorchScript or state_dict .pt)."
        )
        group.add_argument(
            "--meta_path", type=str, default=None, help="ccil_bc_meta.json (state_dict fallback / checks)."
        )
        group.add_argument(
            "--state_key", type=str, default="robot_joint_pos", help="observation['policy'] key fed to the policy."
        )
        group.add_argument("--policy_device", type=str, default="cuda", help="Device for policy tensor operations.")
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> CCILBCPolicy:
        return CCILBCPolicy(CCILBCPolicyConfig.from_cli_args(args))
