# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-reset dome-light property variations.

``DomeLight`` is spawned at a single, non-per-env prim path (``"/World/Light"``, not
``"{ENV_REGEX_NS}/..."``), so unlike per-env asset variations this event ignores
``env_ids`` and resamples the one shared prim whenever it fires at all -- a clean
once-per-episode jitter under ``--num_envs 1`` (the only case this is used for today),
but a single shared value across all envs if this were ever used with ``--num_envs > 1``.
"""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import (
    RunTimeVariationBase,
    VariationBaseCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.assets.object_library import DomeLight


@configclass
class LightPropertyVariationCfg(VariationBaseCfg):
    """Configuration for :class:`LightPropertyVariation`."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[1500.0], high=[4500.0])
    )
    """Uniform distribution over dome-light intensity, resampled every reset."""


class LightPropertyVariation(RunTimeVariationBase):
    """Jitter a :class:`DomeLight`'s intensity every reset.

    Args:
        light: The target dome light.
        cfg: Tunable parameters. Override the intensity distribution via ``cfg.sampler_cfg``.
        name: Identifier under which this variation is registered on the asset.
    """

    cfg: LightPropertyVariationCfg

    def __init__(
        self,
        light: DomeLight,
        cfg: LightPropertyVariationCfg | None = None,
        name: str = "light_intensity",
    ):
        cfg = cfg if cfg is not None else LightPropertyVariationCfg()
        super().__init__(cfg=cfg, name=name)
        self._light = light

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"LightPropertyVariation on '{self._light.name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        event_cfg = EventTermCfg(
            func=randomize_dome_light_intensity,
            mode="reset",
            params={
                "light_prim_path": self._light.prim_path,
                "intensity_sampler": self._sampler,
            },
        )
        return self.name, event_cfg


class randomize_dome_light_intensity(ManagerTermBase):
    """Event term: resample a global dome-light prim's intensity attribute on reset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        light_prim_path: str = cfg.params["light_prim_path"]
        self._prim = env.sim.stage.GetPrimAtPath(light_prim_path)
        assert (
            self._prim.IsValid()
        ), f"randomize_dome_light_intensity: no prim at '{light_prim_path}'."

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        light_prim_path: str,  # noqa: ARG002 (bound to self._prim at __init__ time)
        intensity_sampler: UniformSampler,
    ):
        from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

        if env_ids is None or len(env_ids) == 0:
            return
        intensity = float(intensity_sampler.sample(num_samples=1)[0, 0])
        safe_set_attribute_on_usd_prim(
            self._prim, "inputs:intensity", intensity, camel_case=True
        )


@configclass
class LightColorVariationCfg(VariationBaseCfg):
    """Configuration for :class:`LightColorVariation`."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)

    palette: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (1.0, 0.86, 0.68),
            (0.72, 0.82, 1.0),
            (0.9, 0.9, 0.9),
            (0.78, 1.0, 0.82),
        ]
    )
    """Curated RGB colors (0-1) to choose from each reset."""


class LightColorVariation(RunTimeVariationBase):
    """Jitter a :class:`DomeLight`'s color every reset."""

    cfg: LightColorVariationCfg

    def __init__(
        self,
        light: DomeLight,
        cfg: LightColorVariationCfg | None = None,
        name: str = "light_color",
    ):
        cfg = cfg if cfg is not None else LightColorVariationCfg()
        super().__init__(cfg=cfg, name=name)
        self._light = light

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert (
            self.cfg.palette
        ), f"LightColorVariation on '{self._light.name}': cfg.palette must be non-empty."
        event_cfg = EventTermCfg(
            func=randomize_dome_light_color,
            mode="reset",
            params={
                "light_prim_path": self._light.prim_path,
                "palette": self.cfg.palette,
            },
        )
        return self.name, event_cfg


class randomize_dome_light_color(ManagerTermBase):
    """Event term: resample a global dome-light prim's color attribute on reset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        light_prim_path: str = cfg.params["light_prim_path"]
        self._prim = env.sim.stage.GetPrimAtPath(light_prim_path)
        assert (
            self._prim.IsValid()
        ), f"randomize_dome_light_color: no prim at '{light_prim_path}'."

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        light_prim_path: str,  # noqa: ARG002 (bound to self._prim at __init__ time)
        palette: list[tuple[float, float, float]],
    ):
        from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

        if env_ids is None or len(env_ids) == 0:
            return
        color_index = int(torch.randint(low=0, high=len(palette), size=(1,))[0])
        safe_set_attribute_on_usd_prim(
            self._prim,
            "inputs:color",
            tuple(float(component) for component in palette[color_index]),
            camel_case=True,
        )
