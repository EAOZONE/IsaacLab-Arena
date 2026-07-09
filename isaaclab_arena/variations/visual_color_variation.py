# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-reset discrete visual-color variation.

Wraps :class:`isaaclab.envs.mdp.events.randomize_visual_color`, which only supports a
continuous ``[low_rgb, high_rgb]`` range (any additional entries in ``colors`` are
silently ignored by the installed replicator version). This picks one entry from a
curated palette itself each reset and hands the parent a collapsed ``[color, color]``
range, so ``uniform(low=high)`` deterministically returns exactly that color.
"""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.events import randomize_visual_color
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class VisualColorVariationCfg(VariationBaseCfg):
    """Configuration for :class:`VisualColorVariation`."""

    palette: list[tuple[float, float, float]] = field(default_factory=list)
    """Curated RGB colors (0-1) to choose from each reset. Must be non-empty when enabled."""

    mesh_name: str = ""
    """Prim-path suffix relative to the target asset's ``prim_path``; see ``randomize_visual_color``."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)


class VisualColorVariation(RunTimeVariationBase):
    """Recolor an asset's mesh to a sampler-drawn entry from a curated palette, every reset.

    Args:
        object_name: Scene-entity name of the target asset.
        cfg: Tunable parameters -- set ``cfg.palette`` and ``cfg.mesh_name``.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"{object_name}_color_variation"``.
    """

    cfg: VisualColorVariationCfg

    def __init__(
        self,
        object_name: str,
        cfg: VisualColorVariationCfg | None = None,
        name: str | None = None,
    ):
        cfg = cfg if cfg is not None else VisualColorVariationCfg()
        name = name if name is not None else f"{object_name}_color_variation"
        super().__init__(cfg=cfg, name=name)
        self.object_name = object_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self.cfg.palette, f"VisualColorVariation on '{self.object_name}': cfg.palette must be non-empty."
        assert self._sampler is not None, (
            f"VisualColorVariation on '{self.object_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        event_cfg = EventTermCfg(
            func=randomize_visual_color_choice,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.object_name),
                "mesh_name": self.cfg.mesh_name,
                "palette": self.cfg.palette,
                "sampler": self._sampler,
            },
        )
        return self.name, event_cfg


class randomize_visual_color_choice(randomize_visual_color):
    """Same material/replicator setup as the parent ``__init__``; ``__call__`` picks one palette entry."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        palette: list[tuple[float, float, float]],
        sampler: ChoiceSampler,
        mesh_name: str = "",
    ):
        color = sampler.sample(num_samples=1, choices=palette)[0]
        super().__call__(env, env_ids, event_name="", asset_cfg=asset_cfg, colors=[color, color], mesh_name=mesh_name)
