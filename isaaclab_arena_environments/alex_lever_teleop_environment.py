# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alex teleop environment for the IHMC lever practice board on an 81.28 cm table.

Loads the ``assets/Lever/Levers.usd`` practice board on a visible kinematic
tabletop at 81.28 cm (32 in), with Alex in front for Captury / OpenXR teleop
or closed-loop GR00T eval on the ``H2Ozone/lever_fingers`` / ``H2Ozone/alex_lever`` joint-space
checkpoints.

Normalize the raw export (Y-up inches, no colliders) once per clone::

    /isaac-sim/python.sh isaaclab_arena/scripts/lever/apply_layout_colliders.py

Mount ihmc-alex-sdk inside the container::

    ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk

Record timed teleop demos (no success term — export all takes for manual curation)::

    CAPTURY_HOST=<ip> /isaac-sim/python.sh \\
        isaaclab_arena/scripts/imitation_learning/record_demos.py \\
        --device cuda --viz kit --enable_cameras \\
        --dataset_file /datasets/alex_lever_sim.hdf5 \\
        --num_demos 0 --timed_episode_s 30 \\
        alex_lever_teleop \\
        --teleop_device captury \\
        --embodiment alex_v2_lever_fingers_joint_pos

Replay a recorded LeRobot episode for visual alignment checks::

    /isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/playback_lerobot_dataset.py \\
        --dataset_path datasets/lever_fingers --select_episodes 0 \\
        alex_lever_teleop \\
        --embodiment alex_v2_lever_fingers_joint_pos

**RDX vs Isaac:** IHMC's RDX/SCS2 stack runs momentum-based whole-body control in its
own simulator. This env uses Isaac PhysX with Pink IK (Captury teleop) or joint-position
PD (``alex_v2_lever_fingers_joint_pos`` / GR00T eval). Those are not interchangeable —
see the module docstring on embodiment choice below.
"""

from __future__ import annotations

import argparse
import math
import random
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# IHMC lever-practice table height (81.28 cm) plus a small lift so the board clears the slab.
# Matches procedural_table thickness (4 cm).
_BOARD_VERTICAL_LIFT_M = 0.07
_TABLE_TOP_Z_M = 0.8128 + _BOARD_VERTICAL_LIFT_M
_PROCEDURAL_TABLE_THICKNESS_M = 0.04

# Alex spawn: 0° yaw (Captury torso anchor). Negative X keeps the pelvis clear of the tabletop.
_ALEX_SPAWN_POSE = ((-0.35, 0.0, 0.94296), (0.0, 0.0, 0.0, 1.0))

# Blue_Handled_Valve_v3 handle pose relative to the layout root (board yaw = pi/2).
# Measured with isaaclab_arena/scripts/lever/compute_board_pose.py.
_LEVER3_HANDLE_OFFSET_FROM_LAYOUT = (-0.12274549718409655, -0.02813964244439543, 0.15812658618901088)
# World target for the lever-3 handle: the right index fingertip at the ep-8 grasp frame
# (frame 156 of H2Ozone/alex_lever), so playback lands the demo's hand on the handle.
# This also rests the board plate on the tabletop instead of sinking it ~9 cm in.
_LEVER3_WORLD_TARGET = (0.08387, -0.04905, 0.97803 + _BOARD_VERTICAL_LIFT_M)

# Yaw at which _LEVER3_HANDLE_OFFSET_FROM_LAYOUT was measured; the anchor math rotates
# the offset by (yaw - reference), so any --board_yaw keeps the handle pinned on target.
_OFFSET_MEASUREMENT_YAW = math.pi / 2
# Real board orientation relative to the raw Levers.usd export: the lever cluster faces
# the robot (at yaw pi/2 it sits sideways; at 0 it ends up under the tabletop).
_BOARD_DEFAULT_YAW = math.pi


def _layout_anchor_xy(yaw: float) -> tuple[float, float]:
    """Layout-root world XY that pins the lever-3 handle on _LEVER3_WORLD_TARGET at ``yaw``."""
    delta = yaw - _OFFSET_MEASUREMENT_YAW
    cos_d, sin_d = math.cos(delta), math.sin(delta)
    off_x, off_y = _LEVER3_HANDLE_OFFSET_FROM_LAYOUT[:2]
    return (
        _LEVER3_WORLD_TARGET[0] - (off_x * cos_d - off_y * sin_d),
        _LEVER3_WORLD_TARGET[1] - (off_x * sin_d + off_y * cos_d),
    )


_TABLE_ANCHOR_XY = _layout_anchor_xy(_BOARD_DEFAULT_YAW)
# Extra XY shift on top of the anchor (zero keeps the episode-8 alignment).
_BOARD_DEFAULT_OFFSET_XY = (0.0, 0.0)
# Board mesh centroid in layout-root frame (measured at board_yaw=pi with apply_layout_colliders).
_BOARD_MESH_CENTER_OFFSET_LOCAL = (0.12684999858784548, -0.12068174639013087)

# Per-build placement jitter on the board (seeded), kept small so levers stay reachable.
_LAYOUT_XY_JITTER = 0.0
_LAYOUT_YAW_JITTER = 0.0

# H2Ozone/alex_lever and IHMC hardware logs are 30 Hz. Default Arena is 50 Hz
# (sim.dt=5 ms, decimation=4). Nudge decimation so env.step_dt ≈ 33 ms (~30 Hz).
_ALEX_LEVER_CONTROL_DECIMATION = 6


def _alex_lever_env_cfg_callback(env_cfg):
    env_cfg.decimation = _ALEX_LEVER_CONTROL_DECIMATION
    env_cfg.sim.render_interval = env_cfg.decimation
    return env_cfg

_VALID_ALEX_EMBODIMENTS = (
    "alex_pink",
    "alex_ability_hands",
    "alex_ability_hands_joint_pos",
    "alex_lever_fingers_joint_pos",
    "alex_v2_pink",
    "alex_v2_ability_hands",
    "alex_v2_ability_hands_joint_pos",
    "alex_v2_lever_fingers_joint_pos",
)

_ABILITY_HAND_FINGER_FRICTION_MATERIAL_PATH = "/World/Materials/alex_ability_hand_high_friction_fingers"
_ABILITY_HAND_FINGER_STATIC_FRICTION = 1.5
_ABILITY_HAND_FINGER_DYNAMIC_FRICTION = 1.2
_ABILITY_HAND_FINGER_PRIM_NAME_MARKERS = ("index", "middle", "ring", "pinky", "thumb", "fsr")


@register_environment
class AlexLeverTeleopEnvironment(ExampleEnvironmentBase):
    """Alex + lever practice board on an 81.28 cm table for teleop and joint-space eval."""

    name: str = "alex_lever_teleop"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        assert args_cli.embodiment in _VALID_ALEX_EMBODIMENTS, (
            f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_EMBODIMENTS}"
        )

        rng = random.Random(args_cli.seed)

        layout_yaw = args_cli.board_yaw if args_cli.board_yaw is not None else _BOARD_DEFAULT_YAW
        yaw_anchor_xy = _layout_anchor_xy(layout_yaw)
        table_center_xy = (
            args_cli.table_center_x if args_cli.table_center_x is not None else yaw_anchor_xy[0],
            args_cli.table_center_y if args_cli.table_center_y is not None else yaw_anchor_xy[1],
        )
        board_offset_x = (
            _BOARD_DEFAULT_OFFSET_XY[0] if args_cli.board_offset_x is None else args_cli.board_offset_x
        )
        board_offset_y = (
            _BOARD_DEFAULT_OFFSET_XY[1] if args_cli.board_offset_y is None else args_cli.board_offset_y
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        table = self.asset_registry.get_asset_by_name("lever_practice_table")()
        lever_layout = self.asset_registry.get_asset_by_name("lever_layout")()
        light = self.asset_registry.get_asset_by_name("light")()

        table_center_z = _TABLE_TOP_Z_M - _PROCEDURAL_TABLE_THICKNESS_M / 2.0

        dx = board_offset_x + rng.uniform(-_LAYOUT_XY_JITTER, _LAYOUT_XY_JITTER)
        dy = board_offset_y + rng.uniform(-_LAYOUT_XY_JITTER, _LAYOUT_XY_JITTER)
        dyaw = layout_yaw + rng.uniform(-_LAYOUT_YAW_JITTER, _LAYOUT_YAW_JITTER)
        # Anchor is the layout-root world pose; z pins the lever-3 handle to the
        # real-world target height (the board hangs over the table's near edge).
        layout_xyz = (
            table_center_xy[0] + dx,
            table_center_xy[1] + dy,
            _LEVER3_WORLD_TARGET[2] - _LEVER3_HANDLE_OFFSET_FROM_LAYOUT[2],
        )
        layout_rot = (0.0, 0.0, math.sin(dyaw / 2.0), math.cos(dyaw / 2.0))
        lever_layout.set_initial_pose(Pose(position_xyz=layout_xyz, rotation_xyzw=layout_rot))

        # Tabletop centered under the board mesh (not the layout root or Alex spawn).
        cos_d, sin_d = math.cos(dyaw), math.sin(dyaw)
        bc_x, bc_y = _BOARD_MESH_CENTER_OFFSET_LOCAL
        table.set_initial_pose(
            Pose(
                position_xyz=(
                    layout_xyz[0] + bc_x * cos_d - bc_y * sin_d,
                    layout_xyz[1] + bc_x * sin_d + bc_y * cos_d,
                    table_center_z,
                ),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=_ALEX_SPAWN_POSE[0], rotation_xyzw=_ALEX_SPAWN_POSE[1]))

        if hasattr(embodiment, "set_finger_contact_friction"):
            embodiment.set_finger_contact_friction(
                material_path=_ABILITY_HAND_FINGER_FRICTION_MATERIAL_PATH,
                static_friction=_ABILITY_HAND_FINGER_STATIC_FRICTION,
                dynamic_friction=_ABILITY_HAND_FINGER_DYNAMIC_FRICTION,
                prim_name_markers=_ABILITY_HAND_FINGER_PRIM_NAME_MARKERS,
            )

        teleop_device = None
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        scene = Scene(assets=[ground_plane, table, lever_layout, light])
        env = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            teleop_device=teleop_device,
            env_cfg_callback=_alex_lever_env_cfg_callback,
        )

        light.get_variation("hdr_image").enable()

        return env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None, help="e.g. captury or openxr")
        parser.add_argument(
            "--embodiment",
            type=str,
            default="alex_v2_lever_fingers_joint_pos",
            help=(
                "Alex embodiment. Captury teleop: alex_v2_ability_hands (Pink IK). "
                "GR00T / lever_fingers playback and eval: alex_v2_lever_fingers_joint_pos."
            ),
        )
        parser.add_argument("--seed", type=int, default=0, help="Seed for board placement jitter.")
        parser.add_argument(
            "--table_center_x",
            type=float,
            default=None,
            help=f"Board-placement anchor X (default {_TABLE_ANCHOR_XY[0]}). Tabletop follows the board.",
        )
        parser.add_argument(
            "--table_center_y",
            type=float,
            default=None,
            help=f"Board-placement anchor Y (default {_TABLE_ANCHOR_XY[1]}). Tabletop follows the board.",
        )
        parser.add_argument(
            "--board_offset_x",
            type=float,
            default=None,
            help=f"Board shift along world X (default {_BOARD_DEFAULT_OFFSET_XY[0]}).",
        )
        parser.add_argument(
            "--board_offset_y",
            type=float,
            default=None,
            help=f"Board shift along world Y (default {_BOARD_DEFAULT_OFFSET_XY[1]}).",
        )
        parser.add_argument(
            "--board_yaw",
            type=float,
            default=None,
            help=f"Board yaw about world Z in radians (default {_BOARD_DEFAULT_YAW:.4f} = 90°).",
        )
