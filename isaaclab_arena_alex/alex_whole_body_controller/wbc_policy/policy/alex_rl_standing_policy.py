# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load RSL-RL standing checkpoints for deployment in the WBC action term."""

from __future__ import annotations

import os

import numpy as np
import torch
import warp as wp

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_yaml

from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
    ALEX_STANDING_LOWER_BODY_OFFSET,
    ALEX_STANDING_RL_ACTION_SCALE,
)
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_standing_policy import (
    AlexStandingPolicy,
)


class AlexRLStandingPolicy(AlexStandingPolicy):
    """Deploy an RSL-RL actor checkpoint as the Alex lower-body controller.

    Falls back to :class:`AlexStandingPolicy` when ``model_path`` is unset. Observation
    layout must match :class:`~isaaclab_arena_alex.embodiments.alex_standing_rl.AlexStandingRLObservationsCfg`.
    """

    _warned_fallback = False

    def __init__(
        self,
        *args,
        model_path: str | None = None,
        device: str = "cpu",
        **kwargs,
    ):
        self._model_path = model_path
        self._device = torch.device(device)
        self._actor = None
        self._actor_mlp: torch.nn.Module | None = None
        self._obs_mean: torch.Tensor | None = None
        self._obs_std: torch.Tensor | None = None
        self._default_lower_body: np.ndarray | None = None
        self._rl_action_offset = np.array(ALEX_STANDING_LOWER_BODY_OFFSET, dtype=np.float32)
        super().__init__(*args, **kwargs)
        if model_path:
            self._load_actor(model_path)
        elif not AlexRLStandingPolicy._warned_fallback:
            import warnings

            warnings.warn(
                "Alex RL standing policy has no checkpoint; falling back to classical standing_pd.",
                stacklevel=2,
            )
            AlexRLStandingPolicy._warned_fallback = True

    def _load_actor(self, model_path: str) -> None:
        checkpoint_path = retrieve_file_path(model_path)
        agent_yaml_path = os.path.join(os.path.dirname(checkpoint_path), "params", "agent.yaml")
        agent_cfg = load_yaml(agent_yaml_path) if os.path.exists(agent_yaml_path) else {}

        loaded = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

        if "actor_state_dict" in loaded:
            self._load_actor_state_dict_rsl_rl4(loaded["actor_state_dict"])
            return

        state_dict = loaded.get("model_state_dict", loaded)
        self._load_actor_state_dict_legacy(state_dict, agent_cfg)

    def _load_actor_state_dict_rsl_rl4(self, state_dict: dict) -> None:
        """Load an actor exported by rsl-rl-lib >= 4.x (``actor_state_dict`` with ``mlp.*`` keys)."""
        if "obs_normalizer._mean" in state_dict:
            self._obs_mean = state_dict["obs_normalizer._mean"].to(self._device)
            self._obs_std = state_dict["obs_normalizer._std"].to(self._device)

        mlp_keys = sorted(key for key in state_dict if key.startswith("mlp.") and key.endswith(".weight"))
        assert mlp_keys, f"No mlp.* weights found in actor_state_dict: {list(state_dict)[:12]}"
        layer_indices = [int(key.split(".")[1]) for key in mlp_keys]
        assert layer_indices == list(range(0, layer_indices[-1] + 2, 2)), layer_indices

        layers: list[torch.nn.Module] = []
        for layer_idx in layer_indices:
            weight = state_dict[f"mlp.{layer_idx}.weight"]
            layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
            if layer_idx != layer_indices[-1]:
                layers.append(torch.nn.ELU())

        class _ActorMLP(torch.nn.Module):
            def __init__(self, sequential: torch.nn.Sequential) -> None:
                super().__init__()
                self.mlp = sequential

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                return self.mlp(obs)

        self._actor_mlp = _ActorMLP(torch.nn.Sequential(*layers)).to(self._device)
        self._actor_mlp.load_state_dict(
            {key: value for key, value in state_dict.items() if key.startswith("mlp.")},
            strict=True,
        )
        self._actor_mlp.eval()
        self._actor = None

    def _load_actor_state_dict_legacy(self, state_dict: dict, agent_cfg: dict) -> None:
        """Load an actor exported by older rsl-rl checkpoints (``model_state_dict`` / ``actor.*`` keys)."""
        from rsl_rl.modules import ActorCritic

        actor_weight_keys = sorted(
            (key for key in state_dict if key.startswith("actor.") and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        assert len(actor_weight_keys) >= 2, f"Unexpected actor checkpoint layout: {actor_weight_keys}"
        num_obs = state_dict[actor_weight_keys[0]].shape[1]
        num_actions = state_dict[actor_weight_keys[-1]].shape[0]
        hidden_dims = [state_dict[key].shape[0] for key in actor_weight_keys[:-1]]

        actor_critic = ActorCritic(
            num_obs,
            num_actions,
            init_noise_std=agent_cfg.get("policy", {}).get("init_noise_std", 1.0),
            actor_hidden_dims=hidden_dims,
            critic_hidden_dims=hidden_dims,
            activation=agent_cfg.get("policy", {}).get("activation", "elu"),
        ).to(self._device)
        actor_critic.load_state_dict(state_dict)
        actor_critic.eval()
        self._actor = actor_critic
        self._actor_mlp = None

        if "obs_norm_state_dict" in state_dict:
            obs_norm = state_dict["obs_norm_state_dict"]
            self._obs_mean = obs_norm["_mean"].to(self._device)
            self._obs_std = torch.sqrt(obs_norm["_var"].to(self._device) + obs_norm["eps"])

    def set_default_lower_body_positions(self, positions: np.ndarray) -> None:
        """Cache default lower-body joint positions for action scaling (from the sim asset)."""
        self._default_lower_body = positions.astype(np.float32)

    def _build_policy_observation(self) -> np.ndarray:
        joint_pos = np.asarray(self.observation["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(self.observation["joint_vel"], dtype=np.float32)
        base_ang_vel = np.asarray(self.observation["base_ang_vel"], dtype=np.float32)
        last_action = np.asarray(self.observation.get("last_action", np.zeros_like(joint_pos)), dtype=np.float32)

        if "projected_gravity" in self.observation:
            gravity = np.asarray(self.observation["projected_gravity"], dtype=np.float32)
        else:
            # Back-compat for callers that only pass pelvis orientation.
            from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_standing_policy import (
                gravity_orientation_wxyz,
            )

            pelvis_quat_wxyz = np.asarray(self.observation["pelvis_quat_wxyz"], dtype=np.float32)
            gravity = gravity_orientation_wxyz(pelvis_quat_wxyz)
        return np.concatenate([base_ang_vel, gravity, joint_pos, joint_vel, last_action], axis=1)

    def get_action(self) -> dict[str, object]:
        if self._actor is None and self._actor_mlp is None:
            return super().get_action()

        obs = torch.as_tensor(self._build_policy_observation(), device=self._device, dtype=torch.float32)
        if self._obs_mean is not None and self._obs_std is not None:
            obs = (obs - self._obs_mean) / self._obs_std
        with torch.inference_mode():
            if self._actor_mlp is not None:
                raw_action = self._actor_mlp(obs).cpu().numpy()
            else:
                raw_action = self._actor.act_inference(obs).cpu().numpy()

        offset = self._rl_action_offset
        targets = offset + ALEX_STANDING_RL_ACTION_SCALE * raw_action
        if targets.ndim == 1:
            targets = np.tile(targets, (self.num_envs, 1))
        self.observation["last_action"] = raw_action
        return {"joint_targets": targets.astype(np.float32), "raw_action": raw_action}
