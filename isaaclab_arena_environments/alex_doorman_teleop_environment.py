# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alex teleop environment for opening a variety of DoorMan procedural doors with DR.

Collects a teleop dataset of the IHMC Alex robot opening different doors. Each build loads
one procedurally-generated DoorMan door (articulated, revolute ``hinge_joint``) and applies
domain randomization for sim2real variety:

* **Door variety** — one of the locally-generated DoorMan doors is selected per build
  (``--door_index``, or a seeded random pick when ``--door_index < 0``). An articulation
  cannot be swapped on reset, so door selection is per-build; collect a multi-door dataset by
  looping ``record_demos`` over ``--door_index`` (see below).
* **Lighting** — a :class:`DomeLight` whose HDR environment map is resampled every build via
  Arena's ``hdr_image`` variation.
* **Placement jitter** — the door is offset by a small seeded xy/yaw jitter each build, kept
  within Alex's reach.

The doors are not committed. Generate them once per clone inside the container. This set is
push-open (``--door_open_io out``), latch-free (``--no-build_latch``), and split evenly into
lever and push-bar handles (``--even_handle_split``)::

    /isaac-sim/python.sh isaaclab_arena/scripts/doorman_gen/generate_doors.py \\
        --num_doors 14 --door_open_io out --no-build_latch \\
        --door_handle_type lever pushbar --even_handle_split

(or point ``ARENA_DOORMAN_DOORS_DIR`` at a directory of ``door_NNNN.usd`` files).

Mount the ihmc-alex-sdk so the Alex/ability-hand assets resolve inside the container::

    ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk

Record a multi-door teleop dataset (Captury / OpenXR), appending each door to one HDF5::

    for i in $(seq 0 13); do
      CAPTURY_HOST=<ip> /isaac-sim/python.sh \\
        isaaclab_arena/scripts/imitation_learning/record_demos.py \\
        --device cuda --viz kit --enable_cameras \\
        --dataset_file /datasets/alex_doorman.hdf5 \\
        --num_demos 2 --num_success_steps 10 \\
        alex_doorman_teleop --teleop_device captury \\
        --embodiment alex_v2_ability_hands --door_index $i
    done
"""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Alex spawn: positioned in front of the door, yawed 180° about Z to face it.
_ALEX_SPAWN_POSE = ((0.9, 0.17432, 0.94296), (0.0, 0.0, 1.0, 0.0))
# Base door pose: hinge at the door origin, slab along local +x, lever on the local -y face.
# DoorMan doors load with a different local-frame orientation than ws_alex_door, so 180° about Z
# squares the lever face up to Alex (kept at identity yaw so the Captury torso anchor stays
# aligned), with x tuned so the handle lands at the gripper plane in front of Alex.
_DOORMAN_DOOR_POSE = ((0.34315, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))

# Per-build placement jitter (seeded): small xy offset + yaw wobble, kept within Alex's reach.
_DOOR_XY_JITTER = 0.05
_DOOR_YAW_JITTER = 0.10

_VALID_ALEX_EMBODIMENTS = (
    "alex_pink",
    "alex_ability_hands",
    "alex_ability_hands_joint_pos",
    "alex_v2_pink",
    "alex_v2_ability_hands",
    "alex_v2_ability_hands_joint_pos",
)


@register_environment
class AlexDoormanTeleopEnvironment(ExampleEnvironmentBase):
    """Open one of many DoorMan doors with Alex, with door-variety + HDR + placement DR."""

    name: str = "alex_doorman_teleop"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import math

        from isaaclab_arena.assets.object_library import DoormanDoor, list_doorman_doors
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        assert args_cli.embodiment in _VALID_ALEX_EMBODIMENTS, (
            f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_EMBODIMENTS}"
        )

        rng = random.Random(args_cli.seed)

        # Door variety DR: pick one generated door per build.
        doors = list_doorman_doors()
        num_doors = len(doors)
        assert num_doors > 0, (
            "No DoorMan doors found. Generate them once per clone inside the container with:\n"
            "  /isaac-sim/python.sh isaaclab_arena/scripts/doorman_gen/generate_doors.py --num_doors 15\n"
            "or point ARENA_DOORMAN_DOORS_DIR at a directory of door_NNNN.usd files."
        )
        if args_cli.door_index is not None and args_cli.door_index >= 0:
            assert args_cli.door_index < num_doors, (
                f"--door_index {args_cli.door_index} out of range (only {num_doors} doors generated)."
            )
            door_index = args_cli.door_index
        else:
            door_index = rng.randrange(num_doors)
        door = DoormanDoor(door_index=door_index)
        print(f"[alex_doorman_teleop] door_index={door_index} ({num_doors} doors available)")

        # Ground + door + a (randomizable) dome light. The ground_plane background ships no
        # lights, so the dome light keeps cameras lit (mirrors alex_open_door).
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=_ALEX_SPAWN_POSE[0], rotation_xyzw=_ALEX_SPAWN_POSE[1]))

        # Placement jitter DR: offset the base door pose by a small seeded xy/yaw wobble.
        base_xyz, base_rot = _DOORMAN_DOOR_POSE
        dx = rng.uniform(-_DOOR_XY_JITTER, _DOOR_XY_JITTER)
        dy = rng.uniform(-_DOOR_XY_JITTER, _DOOR_XY_JITTER)
        dyaw = rng.uniform(-_DOOR_YAW_JITTER, _DOOR_YAW_JITTER)
        door_xyz = (base_xyz[0] + dx, base_xyz[1] + dy, base_xyz[2])
        # Compose the base -90° yaw quaternion with the jitter yaw (both about Z).
        base_yaw = 2.0 * math.atan2(base_rot[2], base_rot[3])
        yaw = base_yaw + dyaw
        door_rot = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        door.set_initial_pose(Pose(position_xyz=door_xyz, rotation_xyzw=door_rot))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=[ground_plane, door, light])
        env = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=OpenDoorTask(
                door,
                openness_threshold=args_cli.openness_threshold,
                reset_openness=args_cli.reset_openness,
                episode_length_s=args_cli.episode_length_s,
                task_description=args_cli.task_description,
                fail_on_ik_error=args_cli.fail_on_ik_error,
            ),
            teleop_device=teleop_device,
        )

        # Domain randomization: resample the dome-light HDR each build (visual sim2real).
        light.get_variation("hdr_image").enable()

        return env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--door_index",
            type=int,
            default=-1,
            help="Which generated DoorMan door to load (>=0); negative picks one at random (seeded).",
        )
        parser.add_argument("--seed", type=int, default=0, help="Seed for door selection + placement jitter.")
        parser.add_argument("--teleop_device", type=str, default=None, help="e.g. captury or openxr")
        parser.add_argument("--embodiment", type=str, default="alex_v2_ability_hands")
        parser.add_argument(
            "--fail_on_ik_error",
            action="store_true",
            default=False,
            help="Count an episode as failure if the Pink IK solver fails at any step (even if the door opens).",
        )
        parser.add_argument(
            "--openness_threshold",
            type=float,
            default=0.8,
            help="Door joint percentage above which the episode counts as success.",
        )
        parser.add_argument(
            "--reset_openness",
            type=float,
            default=0.0,
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
