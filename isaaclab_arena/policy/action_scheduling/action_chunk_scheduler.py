# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ActionChunkScheduler: buffer an action chunk and step through it sequentially."""

from __future__ import annotations

import torch
from collections.abc import Callable

from isaaclab_arena.policy.action_scheduling.action_scheduler import ActionScheduler


class ActionChunkScheduler(ActionScheduler):
    """Buffers one action chunk and replays it one step at a time.

    Fetches a new action tensor from the policy only when the current one is exhausted.
    Per-env tracking allows environments to refetch independently.
    """

    def __init__(
        self,
        num_envs: int,
        action_chunk_length: int,
        action_horizon: int,
        action_dim: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_envs = num_envs
        self.action_chunk_length = action_chunk_length
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.device = device

        self.current_action_chunk = torch.zeros(
            (num_envs, action_horizon, action_dim),
            dtype=dtype,
            device=device,
        )
        # Use a bool list to indicate that the action chunk is not yet computed for each env
        # True means the action chunk is not yet computed, False means the action
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.env_requires_new_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Fractional read head into the chunk, in policy-rate (waypoint) units. Advancing it by
        # less than 1.0 per step replays each waypoint for multiple sim steps (zero-order hold),
        # matching an action chunk recorded at a lower rate than the sim runs at. Default 1.0
        # reproduces the original one-waypoint-per-step behavior exactly.
        self._float_index = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self._waypoint_advance: float = 1.0

        # Fetch-efficiency tracking: how many times each env triggered a chunk fetch,
        # and how many envs actually needed the fetch vs. total (wasted compute detection).
        self._n_fetch_calls: int = 0
        self._total_envs_needed: int = 0
        self._per_env_fetch_count = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def get_action(
        self,
        fetch_action_tensor_fn: Callable[[], torch.Tensor],
        hold_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one action per env, refilling the chunk when needed.

        fetch_action_tensor_fn() must return a tensor of shape (num_envs, horizon, action_dim)
        with horizon >= action_horizon.

        ``hold_action`` is part of the base ``ActionScheduler`` API for schedulers that
        need a fallback for waiting envs; this scheduler doesn't have a waiting state
        so the argument is accepted and ignored.
        """
        del hold_action
        if self.env_requires_new_chunk.any():
            new_chunk = fetch_action_tensor_fn()
            mask = self.env_requires_new_chunk
            self.current_action_chunk[mask] = new_chunk[mask]
            self._float_index[mask] = 0.0
            self.env_requires_new_chunk[mask] = False

            self._n_fetch_calls += 1
            n_needed = int(mask.sum().item())
            self._total_envs_needed += n_needed
            self._per_env_fetch_count[mask] += 1

        # Zero-order-hold resample: pick the integer waypoint at the current fractional read
        # head. With the default advance of 1.0 this is just the sequential index; with a
        # smaller advance a waypoint is repeated across sim steps so a low-rate chunk plays at
        # real speed.
        waypoint = torch.floor(self._float_index).to(torch.long).clamp_(0, self.action_horizon - 1)
        batch_idx = torch.arange(self.num_envs, device=self.device)
        action = self.current_action_chunk[batch_idx, waypoint]
        assert action.shape == (
            self.num_envs,
            self.action_dim,
        ), f"{action.shape=} != ({self.num_envs}, {self.action_dim})"

        # Advance the read head; a chunk is exhausted once action_chunk_length waypoints have
        # been consumed (in policy-rate units).
        self._float_index += self._waypoint_advance
        self.env_requires_new_chunk = self._float_index >= self.action_chunk_length

        # Maintain current_action_index in the historical post-increment / -1 convention so
        # consumers (e.g. neck writing via _consumed_chunk_steps) keep working unchanged.
        next_index = (waypoint + 1).to(self.current_action_index.dtype)
        next_index[self.env_requires_new_chunk] = -1
        self.current_action_index = next_index
        self.current_action_chunk[self.env_requires_new_chunk] = 0.0

        return action

    def set_action_rate(self, sim_steps_per_action: float) -> None:
        """Set how many sim steps each chunk waypoint spans (zero-order hold).

        ``sim_steps_per_action = sim_hz / policy_hz`` (e.g. 50/30 ≈ 1.667). A value of 1.0
        restores one-waypoint-per-step behavior.
        """
        assert sim_steps_per_action > 0, "sim_steps_per_action must be positive"
        self._waypoint_advance = 1.0 / sim_steps_per_action

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Reset chunking state for the given envs (all if None)."""
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self._float_index[env_ids] = 0.0
        self.env_requires_new_chunk[env_ids] = True
