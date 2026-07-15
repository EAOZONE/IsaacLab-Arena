# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Empty Alex environment for testing spawn positions.

Just Alex on a bare ground plane — no task objects, no success term. The spawn
pose is set from the CLI so different placements can be tried without editing
code.

View Alex holding still at a given pose (inside the container)::

    /isaac-sim/python.sh tools/view_alex_position.py --viz kit \\
        alex_empty \\
        --embodiment alex_v2_ability_hands \\
        --spawn_pos 0.0,0.0,0.94296 --spawn_yaw 90

Optionally place a USD asset at a trial pose alongside the robot::

    ... alex_empty \\
        --usd isaaclab_arena/assets/lever_sim/Lever_revolute.usd \\
        --usd_pos 0.6,0.0,0.9 --usd_yaw 180

The env also works with the usual runners (teleop.py, record_demos.py, the
GR00T playback script) — pass the same ``--spawn_pos`` / ``--spawn_yaw`` args.
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )

# Default Alex spawn matches alex_teleop_sandbox (lever_eef calibration reference frame).
_DEFAULT_SPAWN_POS = (-0.4, -0.48682, 0.94296)

_VALID_ALEX_EMBODIMENTS = (
    "alex_pink",
    "alex_ability_hands",
    "alex_ability_hands_joint_pos",
    "alex_v2_pink",
    "alex_v2_ability_hands",
    "alex_v2_ability_hands_joint_pos",
)

# Physics rate the base env config runs at (dt=1/200). We keep physics near this and pick an
# integer decimation so the control (env step) rate lands exactly on the requested --control_hz.
_BASE_PHYSICS_HZ = 200.0
_DEFAULT_BACKGROUND_DR_POOL = ("packing_table",)
_LEVER_BLOCKED_BACKGROUNDS = {"kitchen", "kitchen_with_open_drawer"}
_LEVER_NON_CLONEABLE_BACKGROUNDS = {"ground_plane"}
_LIGHT_DR_COLOR_PRESETS = {
    "warm": (1.0, 0.86, 0.68),
    "cool": (0.72, 0.82, 1.0),
    "neutral": (0.9, 0.9, 0.9),
    "greenish": (0.78, 1.0, 0.82),
}


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _unique_preserving_order(values: list[str]) -> list[str]:
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _lever_safe_background_name(background_name: str) -> str:
    if background_name in _LEVER_BLOCKED_BACKGROUNDS:
        return "ground_plane"
    return background_name


def _lever_safe_background_pool(background_names: list[str]) -> list[str]:
    return _unique_preserving_order(
        [
            background_name
            for background_name in background_names
            if background_name
            not in _LEVER_BLOCKED_BACKGROUNDS | _LEVER_NON_CLONEABLE_BACKGROUNDS
        ]
    )


def _parse_light_color_palette(value: str) -> list[tuple[float, float, float]]:
    colors = []
    for token in _parse_csv(value):
        if token in _LIGHT_DR_COLOR_PRESETS:
            colors.append(_LIGHT_DR_COLOR_PRESETS[token])
            continue
        components = [float(component) for component in token.split(":")]
        assert len(components) == 3, (
            f"Invalid light color '{token}'. Use a preset name "
            f"({sorted(_LIGHT_DR_COLOR_PRESETS)}) or r:g:b floats in 0-1."
        )
        colors.append(tuple(components))
    assert colors, "--light_dr_color_palette must contain at least one color."
    return colors


def _make_dr_background(asset_registry, background_name: str, instance_name: str):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType

    source_cls = asset_registry.get_asset_by_name(background_name)
    usd_path = getattr(source_cls, "usd_path", None)
    spawner_cfg = getattr(source_cls, "default_spawner_cfg", None)
    spawner_cfg = spawner_cfg.copy() if spawner_cfg is not None else None
    assert usd_path is not None or spawner_cfg is not None, (
        f"Background '{background_name}' cannot be cloned for DR without instantiating it. "
        "Use a registered background with a class-level usd_path or default_spawner_cfg."
    )
    scale = getattr(source_cls, "scale", (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)
    return Object(
        name=instance_name,
        usd_path=usd_path,
        initial_pose=getattr(source_cls, "initial_pose", None),
        object_type=getattr(source_cls, "object_type", ObjectType.BASE),
        scale=scale,
        spawner_cfg=spawner_cfg,
        spawn_cfg_addon=getattr(source_cls, "spawn_cfg_addon", {}),
        asset_cfg_addon=getattr(source_cls, "asset_cfg_addon", {}),
    )


def _make_control_hz_callback(control_hz: float | None):
    """Build an env_cfg_callback that sets the control (env step) rate to ``control_hz``.

    Returns ``None`` (no override, base 50 Hz) when ``control_hz`` is unset. Physics is kept
    close to the base 200 Hz by choosing the nearest integer decimation, then ``sim.dt`` is set
    so ``control_hz = 1 / (sim.dt * decimation)`` holds exactly. Matching the sim step rate to
    the GR00T policy's ``policy_control_hz`` removes the scheduler's zero-order-hold stretching.
    """
    if control_hz is None:
        return None
    assert control_hz > 0, f"--control_hz must be positive, got {control_hz}"
    decimation = max(1, round(_BASE_PHYSICS_HZ / control_hz))

    def _callback(env_cfg):
        env_cfg.decimation = decimation
        env_cfg.sim.dt = 1.0 / (control_hz * decimation)
        return env_cfg

    return _callback


def _make_env_cfg_callback(
    control_hz: float | None,
    lever_success_object_name: str | None,
    lever_success_angle_deg: float | None,
    robot_dr: bool,
    robot_position_xyz: tuple[float, float, float],
    robot_yaw_rad: float,
    robot_xy_jitter: float,
    robot_yaw_jitter_rad: float,
    background_dr_names: list[str],
):
    control_hz_callback = _make_control_hz_callback(control_hz)
    if (
        control_hz_callback is None
        and lever_success_object_name is None
        and not robot_dr
        and not background_dr_names
    ):
        return None

    def _callback(env_cfg):
        if control_hz_callback is not None:
            env_cfg = control_hz_callback(env_cfg)
        if (
            lever_success_object_name is not None
            and lever_success_angle_deg is not None
        ):
            from isaaclab.managers import TerminationTermCfg

            from isaaclab_arena.tasks.terminations import (
                nested_lever_handle_angle_success,
            )
            from isaaclab_arena_environments.lever_scene_builder import (
                LEVER_HANDLE_RIGID_BODY_SUFFIX,
            )

            env_cfg.terminations.success = TerminationTermCfg(
                func=nested_lever_handle_angle_success,
                params={
                    "object_name": lever_success_object_name,
                    "body_suffix": LEVER_HANDLE_RIGID_BODY_SUFFIX,
                    "angle_threshold_deg": lever_success_angle_deg,
                },
            )
        if robot_dr:
            from isaaclab.managers import EventTermCfg, SceneEntityCfg

            from isaaclab_arena.terms.events import randomize_articulation_root_pose

            env_cfg.events.randomize_alex_root_pose = EventTermCfg(
                func=randomize_articulation_root_pose,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "base_position_xyz": robot_position_xyz,
                    "base_yaw_rad": robot_yaw_rad,
                    "xy_jitter": robot_xy_jitter,
                    "yaw_jitter_rad": robot_yaw_jitter_rad,
                },
            )
        if background_dr_names:
            from isaaclab.managers import EventTermCfg

            from isaaclab_arena.terms.events import randomize_background_visibility

            env_cfg.events.randomize_background_visibility = EventTermCfg(
                func=randomize_background_visibility,
                mode="reset",
                params={"background_names": background_dr_names},
            )
        return env_cfg

    return _callback


@register_environment
class AlexEmptyEnvironment(ExampleEnvironmentBase):
    """Alex alone on a ground plane, spawn pose configurable from the CLI."""

    name: str = "alex_empty"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import lever_scene_builder

        assert (
            args_cli.embodiment in _VALID_ALEX_EMBODIMENTS
        ), f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_EMBODIMENTS}"
        assert (
            len(args_cli.spawn_pos) == 3
        ), f"--spawn_pos needs 3 comma-separated values, got {args_cli.spawn_pos}"

        lever_usd_stem = None
        if args_cli.usd is not None:
            from pathlib import Path

            lever_usd_stem = Path(args_cli.usd).stem.lower()
        is_lever_usd = lever_usd_stem in lever_scene_builder.LEVER_USD_STEMS
        lever_dr_enabled = (
            bool(args_cli.lever_dr) if args_cli.lever_dr is not None else is_lever_usd
        )

        background_name = (
            _lever_safe_background_name(args_cli.background)
            if is_lever_usd
            else args_cli.background
        )
        background_dr_names = []
        scene_assets = []
        if is_lever_usd and background_name == "ground_plane":
            scene_assets.append(self.asset_registry.get_asset_by_name("ground_plane")())
        if lever_dr_enabled and is_lever_usd and args_cli.background_dr_pool != "none":
            background_pool = _lever_safe_background_pool(
                [args_cli.background] + _parse_csv(args_cli.background_dr_pool)
            )
            for index, background_name in enumerate(background_pool):
                instance_name = f"dr_background_{index:02d}_{background_name}"
                scene_assets.append(
                    _make_dr_background(
                        self.asset_registry, background_name, instance_name
                    )
                )
                background_dr_names.append(instance_name)
        else:
            if not (is_lever_usd and background_name == "ground_plane"):
                scene_assets.append(
                    self.asset_registry.get_asset_by_name(background_name)()
                )

        # ground_plane (the default background) ships with no lights, so cameras render the
        # scene as near-black (see e.g. GR00T policy debugging: a vision-conditioned policy
        # trained on real, normally-lit footage sees an out-of-distribution near-black frame
        # and fails to act). Add a dome light, optionally textured with a real-world HDR so
        # camera_obs looks like a lit room rather than a bare grid.
        light = None
        if args_cli.hdr != "none":
            light = self.asset_registry.get_asset_by_name("light")()
            from isaaclab_arena.assets.registries import HDRImageRegistry

            hdr_registry = HDRImageRegistry()
            assert hdr_registry.is_registered(args_cli.hdr), (
                f"--hdr '{args_cli.hdr}' is not a registered HDR "
                f"(available: {sorted(hdr_registry.get_all_keys())}, or 'none' to disable)"
            )
            light.add_hdr(hdr_registry.get_hdr_by_name(args_cli.hdr)())

            if lever_dr_enabled:
                from isaaclab_arena.variations.light_property_variation import (
                    LightColorVariation,
                    LightColorVariationCfg,
                    LightPropertyVariation,
                )

                light.get_variation("hdr_image").enable()
                light.add_variation(LightPropertyVariation(light))
                light.get_variation("light_intensity").enable()
                light.add_variation(
                    LightColorVariation(
                        light,
                        cfg=LightColorVariationCfg(
                            palette=_parse_light_color_palette(
                                args_cli.light_dr_color_palette
                            )
                        ),
                    )
                )
                light.get_variation("light_color").enable()

        if light is not None:
            scene_assets.append(light)
        lever_success_object_name = None
        if args_cli.usd is not None:
            from isaaclab_arena.assets.object import Object

            assert (
                len(args_cli.usd_pos) == 3
            ), f"--usd_pos needs 3 comma-separated values, got {args_cli.usd_pos}"
            usd_stem = lever_usd_stem
            if usd_stem in lever_scene_builder.LEVER_USD_STEMS and tuple(
                args_cli.usd_pos
            ) == (0.6, 0.0, 0.9):
                # Generic default; use the tuned board pose unless the caller overrides it.
                usd_pos = lever_scene_builder.LEVER_USD_DEFAULT_POS
                usd_yaw = (
                    lever_scene_builder.LEVER_USD_DEFAULT_YAW
                    if args_cli.usd_yaw == 0.0
                    else args_cli.usd_yaw
                )
                usd_scale = (
                    lever_scene_builder.LEVER_USD_DEFAULT_SCALE
                    if args_cli.usd_scale == 1.0
                    else args_cli.usd_scale
                )
            else:
                usd_pos = tuple(args_cli.usd_pos)
                usd_yaw = args_cli.usd_yaw
                usd_scale = args_cli.usd_scale

            if usd_stem in lever_scene_builder.LEVER_USD_STEMS:
                lever_assets, lever_object = (
                    lever_scene_builder.build_lever_scene_assets(
                        usd_path=args_cli.usd,
                        usd_pos=usd_pos,
                        usd_yaw=usd_yaw,
                        usd_scale=usd_scale,
                        lever_dr=lever_dr_enabled,
                        table=args_cli.table,
                        lever_dr_xy_jitter=args_cli.lever_dr_xy_jitter,
                        lever_dr_yaw_jitter_deg=args_cli.lever_dr_yaw_jitter_deg,
                    )
                )
                lever_success_object_name = lever_object.name
                scene_assets.extend(lever_assets)
            else:
                usd_initial_pose = Pose(
                    position_xyz=usd_pos, rotation_xyzw=(0.0, 0.70711, 0.70711, 0.0)
                )
                scene_assets.append(
                    Object(
                        name=usd_stem.replace("(", "_").replace(")", "_"),
                        usd_path=args_cli.usd,
                        initial_pose=usd_initial_pose,
                        scale=(usd_scale, usd_scale, usd_scale),
                    )
                )

        lever_success_angle_deg = None
        if lever_success_object_name is not None:
            lever_success_angle_deg = args_cli.lever_success_angle_deg

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras
        )
        half_yaw = math.radians(args_cli.spawn_yaw) / 2.0
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(args_cli.spawn_pos),
                rotation_xyzw=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            )
        )

        teleop_device = None
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(
                args_cli.teleop_device
            )()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=scene_assets),
            teleop_device=teleop_device,
            env_cfg_callback=_make_env_cfg_callback(
                args_cli.control_hz,
                lever_success_object_name,
                lever_success_angle_deg,
                robot_dr=lever_dr_enabled and is_lever_usd,
                robot_position_xyz=tuple(args_cli.spawn_pos),
                robot_yaw_rad=math.radians(args_cli.spawn_yaw),
                robot_xy_jitter=args_cli.robot_dr_xy_jitter,
                robot_yaw_jitter_rad=math.radians(args_cli.robot_dr_yaw_jitter_deg),
                background_dr_names=background_dr_names,
            ),
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="e.g. captury or openxr"
        )
        parser.add_argument("--embodiment", type=str, default="alex_v2_ability_hands")
        parser.add_argument(
            "--spawn_pos",
            type=lambda arg: [float(part) for part in arg.split(",")],
            default=list(_DEFAULT_SPAWN_POS),
            help=f"Robot spawn position x,y,z in world frame (default {','.join(str(v) for v in _DEFAULT_SPAWN_POS)}).",
        )
        parser.add_argument(
            "--spawn_yaw",
            type=float,
            default=0.0,
            help="Robot spawn yaw in degrees about world Z (default 0).",
        )
        parser.add_argument(
            "--background",
            type=str,
            default="ground_plane",
            help="Background asset (default ground_plane).",
        )
        parser.add_argument(
            "--hdr",
            type=str,
            default="home_office_robolab",
            help=(
                "HDR environment map for a dome light (ground_plane ships with no lights, so"
                " cameras otherwise render near-black). Pass 'none' to disable."
            ),
        )
        parser.add_argument(
            "--lever_dr",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Enable domain randomization for lever USD scenes. Defaults to enabled for recognized "
                "lever_sim USDs and disabled for non-lever USDs; pass --no-lever_dr to force fixed setup."
            ),
        )
        parser.add_argument(
            "--lever_dr_xy_jitter",
            type=float,
            default=0.05,
            help="Lever board reset-time xy jitter half-range in metres when lever DR is enabled.",
        )
        parser.add_argument(
            "--lever_dr_yaw_jitter_deg",
            type=float,
            default=25.0,
            help="Lever board reset-time yaw jitter half-range in degrees when lever DR is enabled.",
        )
        parser.add_argument(
            "--robot_dr_xy_jitter",
            type=float,
            default=0.04,
            help="Alex root reset-time xy jitter half-range in metres when lever DR is enabled.",
        )
        parser.add_argument(
            "--robot_dr_yaw_jitter_deg",
            type=float,
            default=8.0,
            help="Alex root reset-time yaw jitter half-range in degrees when lever DR is enabled.",
        )
        parser.add_argument(
            "--background_dr_pool",
            type=str,
            default=",".join(_DEFAULT_BACKGROUND_DR_POOL),
            help=(
                "Comma-separated registered background assets to preload and visibility-swap per reset "
                "when lever DR is enabled. The --background asset is always prepended. Pass 'none' to disable."
            ),
        )
        parser.add_argument(
            "--light_dr_color_palette",
            type=str,
            default="warm,cool,neutral,greenish",
            help=(
                "Comma-separated dome-light color presets or r:g:b values for per-reset light color DR. "
                f"Presets: {','.join(sorted(_LIGHT_DR_COLOR_PRESETS))}."
            ),
        )
        parser.add_argument(
            "--table",
            type=str,
            default="none",
            help=(
                "Workbench placed under a lever_sim board --usd asset (Lever.usd / Lever_revolute.usd),"
                " so it sits on a table instead of floating over the bare ground plane (visual"
                " sim2real). Pass 'none' to disable."
            ),
        )
        parser.add_argument(
            "--usd",
            type=str,
            default=None,
            help="Optional USD file to place in the scene (path valid inside the container).",
        )
        parser.add_argument(
            "--usd_pos",
            type=lambda arg: [float(part) for part in arg.split(",")],
            default=[0.6, 0.0, 0.9],
            help="World position x,y,z for the --usd asset (default 0.6,0.0,0.9).",
        )
        parser.add_argument(
            "--usd_yaw",
            type=float,
            default=0.0,
            help="Yaw in degrees about world Z for the --usd asset (default 0).",
        )
        parser.add_argument(
            "--usd_scale",
            type=float,
            default=1.0,
            help="Uniform scale for the --usd asset (default 1.0; FBX exports are often in cm — try 0.01).",
        )
        parser.add_argument(
            "--control_hz",
            type=float,
            default=None,
            help=(
                "Control (env step) rate in Hz. Default (unset) keeps the base 50 Hz. Set to 30 to "
                "match the GR00T lever_eef policy's policy_control_hz and remove zero-order-hold "
                "stretching (physics stays near 200 Hz via an integer decimation)."
            ),
        )
        parser.add_argument(
            "--lever_success_angle_deg",
            type=float,
            default=20.0,
            help="Lever Handle_1 rotation from reset pose, in degrees, that counts as success for --usd lever scenes.",
        )
