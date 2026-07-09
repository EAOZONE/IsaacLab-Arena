# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-reset dome-light intensity variation.

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

from isaaclab_arena.variations.uniform_sampler import UniformSampler, UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.assets.object_library import DomeLight


@configclass
class LightPropertyVariationCfg(VariationBaseCfg):
    """Configuration for :class:`LightPropertyVariation`."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[1500.0], high=[4500.0]))
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
        assert self._prim.IsValid(), f"randomize_dome_light_intensity: no prim at '{light_prim_path}'."

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
        safe_set_attribute_on_usd_prim(self._prim, "inputs:intensity", intensity, camel_case=True)
