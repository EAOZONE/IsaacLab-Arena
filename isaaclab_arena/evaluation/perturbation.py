# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""External-force perturbations ("pokes") for stress-testing policies during rollout.

A poke applies a constant external wrench to one or more robot links for a short
window of control steps, letting you observe whether a policy recovers after being
bumped off its expected trajectory.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def _quat_from_x_to_dir(direction: torch.Tensor) -> torch.Tensor:
    """Quaternion (x, y, z, w) rotating the arrow's local +X axis onto ``direction``.

    ``direction`` is a length-3 world-frame vector. The arrow USD points along +X, so this
    aligns the rendered arrow with the poke force.
    """
    from isaaclab.utils import math as math_utils

    x_axis = torch.tensor([1.0, 0.0, 0.0], device=direction.device)
    norm = torch.linalg.norm(direction)
    if norm < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=direction.device)
    d = direction / norm
    dot = torch.dot(x_axis, d).clamp(-1.0, 1.0)
    if dot > 1.0 - 1e-6:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=direction.device)  # identity
    if dot < -1.0 + 1e-6:
        return torch.tensor([0.0, 0.0, 1.0, 0.0], device=direction.device)  # 180 deg about z
    axis = torch.linalg.cross(x_axis, d)
    angle = torch.acos(dot)
    return math_utils.quat_from_angle_axis(angle.unsqueeze(0), axis.unsqueeze(0))[0]


class ArmPoke:
    """Apply a constant external wrench to a robot link over a window of control steps.

    The wrench is driven through the articulation's permanent wrench composer, so it is
    applied at every physics substep within the window and is automatically cleared when
    an episode resets. Step indexing is per-episode: the window is defined relative to the
    start of each episode (the counter resets on env reset), so every episode gets poked.
    """

    def __init__(
        self,
        env,
        body: str,
        force: tuple[float, float, float],
        torque: tuple[float, float, float],
        start_step: int,
        duration: int,
        period: int | None,
        is_global: bool,
        robot_name: str = "robot",
        show_marker: bool = True,
    ):
        unwrapped = env.unwrapped
        assert robot_name in unwrapped.scene.articulations, (
            f"Articulation '{robot_name}' not found in scene. Available:"
            f" {list(unwrapped.scene.articulations.keys())}"
        )
        robot: Articulation = unwrapped.scene[robot_name]

        body_ids, body_names = robot.find_bodies(body)
        assert len(body_ids) > 0, (
            f"No body matched '{body}'. Available bodies: {robot.body_names}"
        )

        self._robot = robot
        self._body_ids = list(body_ids)
        self._body_names = body_names
        self._start_step = start_step
        self._duration = duration
        self._period = period
        self._is_global = is_global

        num_envs = unwrapped.num_envs
        num_bodies = len(self._body_ids)
        device = unwrapped.device
        # (num_envs, num_bodies, 3): same wrench broadcast across envs and selected bodies.
        # The "base" tensors are the unscaled wrench; the applied tensors are these times the
        # current per-episode scale (see set_episode_scale), which ramps the poke each episode.
        self._base_forces = torch.tensor(force, dtype=torch.float32, device=device).repeat(num_envs, num_bodies, 1)
        self._base_torques = torch.tensor(torque, dtype=torch.float32, device=device).repeat(num_envs, num_bodies, 1)
        self._zeros = torch.zeros_like(self._base_forces)
        self._scale = 1.0
        self._forces = self._base_forces
        self._torques = self._base_torques
        self._applied = False

        # Optional viewport arrow that lights up at the poked link(s) while the poke is active.
        self._marker = self._build_marker(force, device) if show_marker else None

    def set_episode_scale(self, scale: float) -> None:
        """Scale the base poke force/torque (and arrow length) by ``scale`` for the current episode.

        Used to ramp the poke across episodes — e.g. episode ``k`` (1-based) runs at ``k×`` the
        base wrench to find where the policy stops recovering.
        """
        self._scale = float(scale)
        self._forces = self._base_forces * self._scale
        self._torques = self._base_torques * self._scale

    def _build_marker(self, force: tuple[float, float, float], device):
        """Create a red arrow marker per (env, poked body), oriented along the force direction.

        The arrow's local +X is aligned with the world-frame force vector and its length is
        scaled by the force magnitude. Direction is exact for world-frame pokes (the default);
        for body-frame pokes it shows the force as a world vector and ignores subsequent body
        rotation, which is fine as a "poke happening here" indicator.
        """
        import copy

        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

        force_t = torch.tensor(force, dtype=torch.float32, device=device)
        force_mag = float(torch.linalg.norm(force_t))
        # ~0.5 m arrow for a 40 N poke, clamped so it stays visible but not huge.
        length = min(max(force_mag * 0.0125, 0.2), 2.0)
        self._marker_dir = force_t / force_mag if force_mag > 1e-8 else force_t

        cfg = copy.deepcopy(RED_ARROW_X_MARKER_CFG)
        cfg.prim_path = "/Visuals/poke_arrow"
        base_scale = cfg.markers["arrow"].scale  # (length, thickness, thickness)
        cfg.markers["arrow"].scale = (length, base_scale[1], base_scale[2])
        # Base (episode-scale 1) arrow length; _update_marker grows it with self._scale.
        self._marker_base_length = length
        self._marker_thickness = base_scale[1]

        num_markers = self._base_forces.shape[0] * len(self._body_ids)  # num_envs * num_bodies
        quat = _quat_from_x_to_dir(self._marker_dir)
        self._marker_quats = quat.unsqueeze(0).repeat(num_markers, 1)

        marker = VisualizationMarkers(cfg)
        marker.set_visibility(False)
        return marker

    def _update_marker(self) -> None:
        """Place the arrow(s) so the tip sits on each poked body, pointing along the force.

        The arrow length grows with the current episode scale so a bigger poke draws a bigger
        arrow.
        """
        # (num_envs, num_bodies, 3) -> (num_envs * num_bodies, 3)
        body_pos = self._robot.data.body_link_pos_w
        if not isinstance(body_pos, torch.Tensor):
            import warp as wp

            body_pos = wp.to_torch(body_pos)
        body_pos = body_pos[:, self._body_ids].reshape(-1, 3)
        length = self._marker_base_length * self._scale
        # Offset the arrow tail back along the force so the arrowhead lands on the link.
        translations = body_pos - self._marker_dir * length
        scales = self._marker_quats.new_tensor([length, self._marker_thickness, self._marker_thickness])
        scales = scales.unsqueeze(0).repeat(self._marker_quats.shape[0], 1)
        self._marker.visualize(translations=translations, orientations=self._marker_quats, scales=scales)

    def _is_active(self, episode_step: int) -> bool:
        if episode_step < self._start_step:
            return False
        if self._period is not None:
            return (episode_step - self._start_step) % self._period < self._duration
        return (episode_step - self._start_step) < self._duration

    def apply(self, episode_step: int) -> bool:
        """Set (or clear) the poke wrench for the given per-episode step.

        Returns True when the poke is actively pushing on this step.
        """
        active = self._is_active(episode_step)
        # Only touch the sim on the active steps and on the first step after the window
        # ends (to clear the wrench); otherwise leave it untouched.
        if active:
            self._robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self._forces,
                torques=self._torques,
                body_ids=self._body_ids,
                is_global=self._is_global,
            )
        elif self._applied:
            self._robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self._zeros,
                torques=self._zeros,
                body_ids=self._body_ids,
                is_global=self._is_global,
            )
        if self._marker is not None:
            if active:
                self._marker.set_visibility(True)
                self._update_marker()
            elif self._applied:
                self._marker.set_visibility(False)
        self._applied = active
        return active

    @property
    def body_names(self) -> list[str]:
        return self._body_names
