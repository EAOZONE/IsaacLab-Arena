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

import math
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

    With ``random_force=True`` the poke becomes a *random nudge*: at the start of every
    episode (see :meth:`resample`) each env independently draws a random horizontal
    direction and a magnitude sampled uniformly from ``force_range``, so the policy is
    bumped a different way each rollout. Random pokes are not ramped per episode (the
    deterministic ``set_episode_scale`` ramp is disabled); the randomness supplies the
    variation instead.
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
        random_force: bool = False,
        force_range: tuple[float, float] = (20.0, 45.0),
        seed: int | None = None,
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

        # Random-nudge config: when enabled, resample() redraws a horizontal push per env
        # each episode. A dedicated generator keeps pokes reproducible (and independent of
        # the global RNG) when a seed is given.
        self._random_force = random_force
        self._force_min, self._force_max = float(force_range[0]), float(force_range[1])
        self._generator = None
        if random_force:
            self._generator = torch.Generator(device=device)
            if seed is not None:
                self._generator.manual_seed(int(seed))
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

        # Draw an initial random push so the poke is well-defined before the first episode
        # (the runner also calls resample() at each episode boundary). No-op when not random.
        self.resample()

    def set_episode_scale(self, scale: float) -> None:
        """Scale the base poke force/torque (and arrow length) by ``scale`` for the current episode.

        Used to ramp the poke across episodes — e.g. episode ``k`` (1-based) runs at ``k×`` the
        base wrench to find where the policy stops recovering. Disabled for random pokes, whose
        magnitude is resampled per episode instead of ramped.
        """
        if self._random_force:
            return
        self._scale = float(scale)
        self._forces = self._base_forces * self._scale
        self._torques = self._base_torques * self._scale

    def resample(self) -> None:
        """Draw a fresh random horizontal push (direction + magnitude) for every env.

        No-op unless constructed with ``random_force=True``. Called at the start of each
        episode so every episode — and every parallel env — is nudged in a different random
        horizontal (world-frame xy) direction, with a magnitude sampled uniformly from the
        configured ``force_range``.
        """
        if not self._random_force:
            return
        num_envs, num_bodies = self._base_forces.shape[0], self._base_forces.shape[1]
        device = self._base_forces.device
        gen = self._generator
        theta = torch.rand(num_envs, device=device, generator=gen) * (2.0 * math.pi)
        mag = self._force_min + torch.rand(num_envs, device=device, generator=gen) * (
            self._force_max - self._force_min
        )
        dirs = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)  # (num_envs, 3)
        forces = dirs * mag.unsqueeze(-1)  # (num_envs, 3)
        self._base_forces = forces.unsqueeze(1).repeat(1, num_bodies, 1)
        # Random pokes are not ramped, so the applied wrench is just the freshly drawn one.
        self._forces = self._base_forces
        self._torques = self._base_torques
        if self._marker is not None:
            per_marker_dirs = dirs.unsqueeze(1).repeat(1, num_bodies, 1).reshape(-1, 3)  # (num_markers, 3)
            self._marker_dirs = per_marker_dirs
            self._marker_quats = torch.stack([_quat_from_x_to_dir(d) for d in per_marker_dirs])
            per_marker_mag = mag.unsqueeze(1).repeat(1, num_bodies).reshape(-1)
            self._marker_lengths = torch.tensor(
                [self._length_for_mag(float(m)) for m in per_marker_mag], device=device
            )

    @staticmethod
    def _length_for_mag(force_mag: float) -> float:
        """Arrow length [m] for a force magnitude: ~0.5 m at 40 N, clamped to stay visible."""
        return min(max(force_mag * 0.0125, 0.2), 2.0)

    def _build_marker(self, force: tuple[float, float, float], device):
        """Create a red arrow marker per (env, poked body), oriented along the force direction.

        The arrow's local +X is aligned with the world-frame force vector and its length is
        scaled by the force magnitude. Direction is exact for world-frame pokes (the default);
        for body-frame pokes it shows the force as a world vector and ignores subsequent body
        rotation, which is fine as a "poke happening here" indicator.

        Direction/length/orientation are stored as per-marker tensors (one row per env×body).
        For a deterministic poke every row is identical; :meth:`resample` overwrites them per
        env for a random poke.
        """
        import copy

        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

        force_t = torch.tensor(force, dtype=torch.float32, device=device)
        force_mag = float(torch.linalg.norm(force_t))
        direction = force_t / force_mag if force_mag > 1e-8 else force_t

        cfg = copy.deepcopy(RED_ARROW_X_MARKER_CFG)
        cfg.prim_path = "/Visuals/poke_arrow"
        self._marker_thickness = cfg.markers["arrow"].scale[1]  # (length, thickness, thickness)

        num_markers = self._base_forces.shape[0] * len(self._body_ids)  # num_envs * num_bodies
        # Per-marker base (episode-scale 1) direction, orientation and length; _update_marker
        # grows the length with self._scale.
        self._marker_dirs = direction.unsqueeze(0).repeat(num_markers, 1)
        self._marker_quats = _quat_from_x_to_dir(direction).unsqueeze(0).repeat(num_markers, 1)
        self._marker_lengths = torch.full((num_markers,), self._length_for_mag(force_mag), device=device)

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
        lengths = self._marker_lengths * self._scale  # (num_markers,)
        # Offset each arrow tail back along its force so the arrowhead lands on the link.
        translations = body_pos - self._marker_dirs * lengths.unsqueeze(-1)
        thickness = lengths.new_full(lengths.shape, self._marker_thickness)
        scales = torch.stack([lengths, thickness, thickness], dim=-1)
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
