# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""View Alex holding still in an environment, for testing spawn positions.

Steps the env with wrist targets held at the current EEF poses (no teleop, no
policy) so the robot stands in place while you inspect its placement in the GUI.

Run inside the container:

    /isaac-sim/python.sh tools/view_alex_position.py --viz kit \\
        alex_empty \\
        --embodiment alex_v2_ability_hands \\
        --spawn_pos 0.0,0.0,0.94296 --spawn_yaw 90

Requires an ability-hands embodiment (the 34-dim EEF action space).
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--steps", type=int, default=0, help="Number of steps to run; 0 runs until the app is closed.")
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import torch

from isaaclab_arena.embodiments.alex.alex import (
    ALEX_ABILITY_HAND_WRIST_ACTION_DIM,
    stabilize_alex_ability_hand_teleop_action,
)


def main() -> None:
    arena_builder = get_arena_builder_from_cli(args_cli)
    env = arena_builder.make_registered()
    env.reset()

    robot = env.unwrapped.scene["robot"]
    action_dim = env.action_space.shape[-1]
    assert action_dim >= ALEX_ABILITY_HAND_WRIST_ACTION_DIM, (
        f"Hold-still viewing needs the ability-hands wrist action space"
        f" (>= {ALEX_ABILITY_HAND_WRIST_ACTION_DIM} dims), got {action_dim}"
    )

    pelvis_idx = robot.body_names.index("PELVIS_LINK")
    step = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            action = torch.zeros(action_dim, device=env.unwrapped.device)
            action = stabilize_alex_ability_hand_teleop_action(env.unwrapped, action, force_hold_wrists=True)
            env.step(action.repeat(env.unwrapped.num_envs, 1))

            if step == 10:
                import warp as wp

                pelvis_pos = wp.to_torch(robot.data.body_pos_w)[0, pelvis_idx]
                print(f"[view] settled pelvis position: {pelvis_pos.tolist()}", flush=True)
            step += 1
            if args_cli.steps > 0 and step >= args_cli.steps:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
