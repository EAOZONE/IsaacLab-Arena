# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class AlexWBCPolicy(ABC):
    """Base class for Alex lower-body whole-body controllers."""

    def set_goal(self, goal: dict[str, object]) -> None:
        """Set optional high-level commands (reserved for Phase-1 RL / locomotion)."""

    def set_observation(self, observation: dict[str, object]) -> None:
        """Update the policy with the latest simulator state."""
        self.observation = observation

    @abstractmethod
    def get_action(self) -> dict[str, object]:
        """Return lower-body joint position targets keyed by ``joint_targets``."""

    def reset(self, env_ids: object | None = None) -> None:
        """Reset internal state for the given environment indices."""

    def close(self) -> None:
        """Release resources held by the policy."""
