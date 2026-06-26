# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CLI helpers for Alex WBC embodiments (classical or RL lower-body standing)."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_WBC_VERSION_RL,
    ALEX_WBC_VERSION_STANDING_PD,
)

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

ALEX_WBC_EMBODIMENT_NAMES: frozenset[str] = frozenset(
    {
        "alex_wbc_ability_hands",
        "alex_v2_wbc_ability_hands",
        "alex_wbc_pink",
        "alex_v2_wbc_pink",
    }
)


def add_alex_standing_wbc_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register optional lower-body standing policy flags for WBC Alex embodiments."""
    group = parser.add_argument_group(
        "Alex standing WBC",
        "Lower-body controller for floating-pelvis WBC embodiments (ignored for fixed-base teleop).",
    )
    group.add_argument(
        "--standing_wbc_version",
        type=str,
        default=None,
        choices=[ALEX_WBC_VERSION_STANDING_PD, ALEX_WBC_VERSION_RL],
        help="Lower-body backend: classical PD (default) or trained RL policy.",
    )
    group.add_argument(
        "--standing_model_path",
        type=str,
        default=None,
        help="RSL-RL checkpoint for --standing_wbc_version=rl (e.g. logs/rsl_rl/alex_standing_balance/.../model_2999.pt).",
    )


def alex_standing_wbc_kwargs_from_cli(args_cli: argparse.Namespace) -> dict[str, str]:
    """Return embodiment constructor kwargs for standing WBC, or an empty dict."""
    if args_cli.embodiment not in ALEX_WBC_EMBODIMENT_NAMES:
        return {}

    kwargs: dict[str, str] = {}
    if args_cli.standing_wbc_version is not None:
        kwargs["standing_wbc_version"] = args_cli.standing_wbc_version
    if args_cli.standing_model_path is not None:
        kwargs["standing_model_path"] = args_cli.standing_model_path
    return kwargs


def apply_standing_wbc_to_action_config(
    action_config,
    *,
    standing_wbc_version: str | None = None,
    standing_model_path: str | None = None,
) -> None:
    """Configure ``lower_body_standing`` on a WBC action config."""
    if standing_wbc_version is None and standing_model_path is None:
        return

    if standing_model_path is not None and standing_wbc_version is None:
        standing_wbc_version = ALEX_WBC_VERSION_RL

    assert hasattr(action_config, "lower_body_standing"), (
        "action_config must expose lower_body_standing (WBC embodiment action cfg)"
    )
    action_config.lower_body_standing.wbc_version = standing_wbc_version or ALEX_WBC_VERSION_STANDING_PD
    if standing_model_path is not None:
        action_config.lower_body_standing.model_path = standing_model_path


def build_alex_embodiment(asset_registry, args_cli: argparse.Namespace) -> EmbodimentBase:
    """Instantiate an Alex embodiment, forwarding standing-WBC CLI kwargs when applicable."""
    kwargs = {"enable_cameras": args_cli.enable_cameras}
    kwargs.update(alex_standing_wbc_kwargs_from_cli(args_cli))
    return asset_registry.get_asset_by_name(args_cli.embodiment)(**kwargs)
