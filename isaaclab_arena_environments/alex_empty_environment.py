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
        --usd_pos 0.6,0.0,0.9 --usd_yaw 90

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

# Tuned lever-board pose when --usd points at one of the lever_sim board USDs (see
# Pictures/Screenshots 2026-07-04). Lever_revolute.usd (2026-07-07) shares the same
# Layout_v9 origin and inches units (mpu=0.0254) as Lever.usd, just with its physics
# authored as a single dynamic rigid body (the handle) jointed straight to the static
# base instead of the fragile ArticulationRootAPI + dummy-link workaround in
# Lever_physics.usd, so the same tuned pose/scale applies.
_LEVER_USD_STEMS = ("lever", "lever_revolute")
_LEVER_USD_DEFAULT_POS = (-0.05062, -0.51385, 0.75167)
_LEVER_USD_DEFAULT_YAW = 90.0
_LEVER_USD_DEFAULT_SCALE = 0.0254

# Workbench placed under the lever board (visual sim2real: the real lever_eef dataset was
# recorded with the fixture bolted to a wooden bench, not floating over a bare grid floor).
# SeattleLabTable's own prim origin sits ~(0.37, 0.16) away from its mesh center in its local
# xy (measured via UsdGeom.BBoxCache), so it's placed at the lever xy minus that offset to
# actually center the tabletop under the lever. z is tuned so its surface meets the lever base.
_LEVER_TABLE_XY_OFFSET = (0.37025, 0.15521)
_LEVER_TABLE_POS_Z = 0.0

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

        assert (
            args_cli.embodiment in _VALID_ALEX_EMBODIMENTS
        ), f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_EMBODIMENTS}"
        assert (
            len(args_cli.spawn_pos) == 3
        ), f"--spawn_pos needs 3 comma-separated values, got {args_cli.spawn_pos}"

        background = self.asset_registry.get_asset_by_name(args_cli.background)()

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

        scene_assets = [background]
        if light is not None:
            scene_assets.append(light)
        if args_cli.usd is not None:
            from pathlib import Path

            from isaaclab_arena.assets.object import Object

            assert (
                len(args_cli.usd_pos) == 3
            ), f"--usd_pos needs 3 comma-separated values, got {args_cli.usd_pos}"
            usd_stem = Path(args_cli.usd).stem.lower()
            if usd_stem in _LEVER_USD_STEMS and tuple(args_cli.usd_pos) == (0.6, 0.0, 0.9):
                # Generic default; use the tuned board pose unless the caller overrides it.
                usd_pos = _LEVER_USD_DEFAULT_POS
                usd_yaw = (
                    _LEVER_USD_DEFAULT_YAW
                    if args_cli.usd_yaw == 0.0
                    else args_cli.usd_yaw
                )
                usd_scale = (
                    _LEVER_USD_DEFAULT_SCALE
                    if args_cli.usd_scale == 1.0
                    else args_cli.usd_scale
                )
            else:
                usd_pos = tuple(args_cli.usd_pos)
                usd_yaw = args_cli.usd_yaw
                usd_scale = args_cli.usd_scale
            usd_half_yaw = math.radians(usd_yaw) / 2.0
            scene_assets.append(
                Object(
                    name=usd_stem.replace("(", "_").replace(")", "_"),
                    usd_path=args_cli.usd,
                    initial_pose=Pose(
                        position_xyz=usd_pos,
                        rotation_xyzw=(0.0, 0.70711, 0.70711, 0.0),
                    ),
                    scale=(usd_scale, usd_scale, usd_scale),
                )
            )
            if usd_stem in _LEVER_USD_STEMS and args_cli.table != "none":
                from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

                scene_assets.append(
                    Object(
                        name="lever_table",
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                        initial_pose=Pose(
                            position_xyz=(
                                usd_pos[0] - _LEVER_TABLE_XY_OFFSET[0],
                                usd_pos[1] - _LEVER_TABLE_XY_OFFSET[1],
                                _LEVER_TABLE_POS_Z,
                            ),
                            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                        ),
                    )
                )

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
            env_cfg_callback=_make_control_hz_callback(args_cli.control_hz),
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
            "--table",
            type=str,
            default="seattle_lab",
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
