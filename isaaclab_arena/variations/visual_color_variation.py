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
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import (
    RunTimeVariationBase,
    VariationBaseCfg,
)

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
        assert (
            self.cfg.palette
        ), f"VisualColorVariation on '{self.object_name}': cfg.palette must be non-empty."
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


class randomize_visual_color_choice(ManagerTermBase):
    """Randomize visual color from a discrete palette.

    Rigid objects and articulations delegate to Isaac Lab's Replicator-backed visual-color
    event. Base assets are represented in the scene as ``XformPrimView`` objects and do not
    have ``asset.cfg``, so they need their mesh paths resolved directly from ``prim_paths``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        mesh_name: str = cfg.params.get("mesh_name", "")
        asset = env.scene[asset_cfg.name]
        self._delegate = None
        self._shader_diffuse_color_attrs = []

        if hasattr(asset, "cfg"):
            self._delegate = randomize_visual_color(cfg, env)
            return

        from pxr import Sdf, UsdShade

        if not mesh_name.startswith("/"):
            mesh_name = "/" + mesh_name
        prim_paths = getattr(asset, "prim_paths", None)
        assert prim_paths is not None, (
            f"randomize_visual_color_choice expects scene['{asset_cfg.name}'] to expose "
            "either .cfg or .prim_paths; got "
            f"{type(asset).__name__}."
        )
        mesh_prims = []
        for asset_prim_path in prim_paths:
            mesh_prim = env.sim.stage.GetPrimAtPath(f"{asset_prim_path}{mesh_name}")
            assert mesh_prim.IsValid(), (
                "randomize_visual_color_choice could not find mesh prim "
                f"'{asset_prim_path}{mesh_name}'."
            )
            if mesh_prim.IsInstanceable():
                mesh_prim.SetInstanceable(False)
            mesh_prims.append(mesh_prim)

        looks_scope = "/World/Looks"
        env.sim.stage.DefinePrim(looks_scope, "Scope")
        for index, mesh_prim in enumerate(mesh_prims):
            material_path = (
                f"{looks_scope}/{asset_cfg.name}_{index}_color_variation_mat"
            )
            shader_path = f"{material_path}/Shader"
            material = UsdShade.Material.Define(env.sim.stage, material_path)
            shader = UsdShade.Shader.Define(env.sim.stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            diffuse_input = shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)
            self._shader_diffuse_color_attrs.append(diffuse_input)

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
        if self._delegate is not None:
            self._delegate(
                env,
                env_ids,
                event_name="",
                asset_cfg=asset_cfg,
                colors=[color, color],
                mesh_name=mesh_name,
            )
            return

        if env_ids is None or len(env_ids) == 0:
            return
        from pxr import Gf

        usd_color = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        for attr in self._shader_diffuse_color_attrs:
            attr.Set(usd_color)
