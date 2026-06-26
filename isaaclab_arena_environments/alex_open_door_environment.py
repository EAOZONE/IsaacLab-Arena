# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alex teleop environment for opening a hinged door.

Task: reach the door handle and swing the door open past the success threshold. Uses
:class:`~isaaclab_arena.tasks.open_door_task.OpenDoorTask` on a revolute-hinge asset.
Two doors are available via ``--door``:

* ``ws_alex_door`` (default) — the ws_alex (IHMC) full-size door, rebuilt as a single-DOF
  articulation by ``isaaclab_arena/scripts/doorman/build_ws_alex_door.py``. The generated
  USD is not committed; build it once inside the container before first use::

      /isaac-sim/python.sh isaaclab_arena/scripts/doorman/build_ws_alex_door.py --headless

* ``microwave`` — the Lightwheel microwave appliance door (revolute ``microjoint``).

Designed for Captury mocap teleop with identity robot yaw so torso anchoring stays
aligned (contrast with the RoboCasa fridge env, which spawns Alex turned +90°).

Mount ihmc-alex-sdk inside the container::

    ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk

Record demos::

    CAPTURY_HOST=<ip> CAPTURY_VISUALIZE_SKELETON=1 \\
    /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/record_demos.py \\
        --device cuda --viz kit --enable_cameras \\
        --dataset_file /datasets/alex_open_door.hdf5 \\
        --num_demos 5 --num_success_steps 10 \\
        alex_open_door \\
        --teleop_device captury \\
        --embodiment alex_v2_ability_hands
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Alex spawn tuned for the kitchen + microwave layout (0° yaw — Captury-friendly).
_ALEX_SPAWN_POSE = ((-0.4, -0.48682, 0.94296), (0.0, 0.0, 0.0, 1.0))
# Microwave on the kitchen counter (same tuned pose as ``alex_open_microwave``).
_MICROWAVE_POSE = ((0.4, -0.00586, 0.22773), (0.0, 0.0, -0.7071068, 0.7071068))
# ws_alex full-size door: hinge at the door origin, slab extends along local +x, lever on
# the local -y face. Rotated -90° about Z so the lever face points back toward Alex (-x).
# Placed so the lever lands ~0.27 m forward, ~0.66 m from the spawn base, at handle height,
# and the panel is the wall directly ahead of Alex (push-to-open, hinged on Alex's left).
_WS_ALEX_DOOR_POSE = ((0.0, 0.0, 0.0), (0.0, 0.0, -0.7071068, 0.7071068))

_VALID_ALEX_EMBODIMENTS = (
    "alex_pink",
    "alex_ability_hands",
    "alex_ability_hands_joint_pos",
    "alex_wbc_pink",
    "alex_wbc_ability_hands",
    "alex_v2_pink",
    "alex_v2_ability_hands",
    "alex_v2_ability_hands_joint_pos",
    "alex_v2_wbc_pink",
    "alex_v2_wbc_ability_hands",
)

_SUPPORTED_DOORS = ("ws_alex_door", "microwave")

# Sensible background per door when ``--background`` is not given explicitly.
_DEFAULT_BACKGROUND = {"ws_alex_door": "ground_plane", "microwave": "kitchen"}


@register_environment
class AlexOpenDoorEnvironment(ExampleEnvironmentBase):
    """Open a hinged door (microwave) with Alex and Captury or OpenXR teleop."""

    name: str = "alex_open_door"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_alex.embodiments.alex_wbc_cli import build_alex_embodiment

        assert args_cli.embodiment in _VALID_ALEX_EMBODIMENTS, (
            f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_EMBODIMENTS}"
        )
        assert args_cli.door in _SUPPORTED_DOORS, f"Unsupported door '{args_cli.door}'; choose one of {_SUPPORTED_DOORS}"

        background_name = args_cli.background or _DEFAULT_BACKGROUND[args_cli.door]
        background = self.asset_registry.get_asset_by_name(background_name)()
        door = self._make_door(args_cli.door)
        # Add a dome light so the cameras are not black: the default ws_alex_door
        # background (ground_plane) ships no lights. Mirrors alex_put_and_close_door
        # and the table-top environments; harmless extra fill for the kitchen background.
        light = self.asset_registry.get_asset_by_name("light")()
        assets = [background, door, light]

        embodiment = build_alex_embodiment(self.asset_registry, args_cli)
        spawn_xyz, spawn_rot = _ALEX_SPAWN_POSE
        embodiment.set_initial_pose(Pose(position_xyz=spawn_xyz, rotation_xyzw=spawn_rot))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        door_xyz, door_rot = _MICROWAVE_POSE if args_cli.door == "microwave" else _WS_ALEX_DOOR_POSE
        door.set_initial_pose(Pose(position_xyz=door_xyz, rotation_xyzw=door_rot))

        scene = Scene(assets=assets)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=OpenDoorTask(
                door,
                openness_threshold=args_cli.openness_threshold,
                reset_openness=args_cli.reset_openness,
                episode_length_s=args_cli.episode_length_s,
                task_description=args_cli.task_description,
            ),
            teleop_device=teleop_device,
        )

    def _make_door(self, door_name: str):
        if door_name in _SUPPORTED_DOORS:
            return self.asset_registry.get_asset_by_name(door_name)()
        raise ValueError(f"Unsupported door: {door_name}")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--door",
            type=str,
            default="ws_alex_door",
            choices=_SUPPORTED_DOORS,
            help="Which hinged door to open: the ws_alex full-size door or the microwave.",
        )
        parser.add_argument(
            "--background",
            type=str,
            default=None,
            help="Background scene; defaults per door (ground_plane for ws_alex_door, kitchen for microwave).",
        )
        parser.add_argument("--teleop_device", type=str, default=None, help="e.g. captury or openxr")
        parser.add_argument("--embodiment", type=str, default="alex_v2_ability_hands")
        from isaaclab_arena_alex.embodiments.alex_wbc_cli import add_alex_standing_wbc_cli_args

        add_alex_standing_wbc_cli_args(parser)
        parser.add_argument(
            "--openness_threshold",
            type=float,
            default=0.8,
            help="Door joint percentage above which the episode counts as success.",
        )
        parser.add_argument(
            "--reset_openness",
            type=float,
            default=0.2,
            help="Door starts this fraction open (0=closed, 1=fully open).",
        )
        parser.add_argument(
            "--episode_length_s",
            type=float,
            default=20.0,
            help="Max episode duration [s] before timeout.",
        )
        parser.add_argument(
            "--task_description",
            type=str,
            default="Reach the door handle and swing the door open.",
        )
