# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: Alex WBC embodiment stays upright with wrists held still (no fixed pelvis)."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_alex_wbc_standing_hold(simulation_app) -> bool:
    import gymnasium as gym
    import torch
    import warp as wp

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.embodiments.alex.alex import stabilize_alex_ability_hand_teleop_action
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.alex_teleop_sandbox_environment import AlexTeleopSandboxEnvironment

    args = get_isaaclab_arena_cli_parser().parse_args([])
    args.embodiment = "alex_wbc_ability_hands"
    args.background = "ground_plane"
    args.teleop_device = None
    args.enable_cameras = False

    arena_env = AlexTeleopSandboxEnvironment().get_env(args)
    builder = ArenaEnvBuilder(arena_env, args)
    if arena_env.name in gym.registry:
        del gym.registry[arena_env.name]
    env = builder.make_registered()
    env.reset()

    robot = env.unwrapped.scene["robot"]
    action_dim = env.action_space.shape[-1]
    pelvis_idx = robot.body_names.index("PELVIS_LINK")

    assert robot.cfg.spawn.fix_base is False, "WBC embodiment must use a floating pelvis"

    max_pelvis_drift = 0.0
    for _ in range(80):
        action = torch.zeros(action_dim, device=env.unwrapped.device)
        action = stabilize_alex_ability_hand_teleop_action(env.unwrapped, action, force_hold_wrists=True)
        actions = action.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
        with torch.inference_mode():
            env.step(actions)

        body_pos = wp.to_torch(robot.data.body_pos_w)[0]
        pelvis_z = float(body_pos[pelvis_idx, 2].item())
        max_pelvis_drift = max(max_pelvis_drift, abs(pelvis_z - 0.93))

    env.close()
    # Allow modest settling from the nominal crouch; failure mode is launch / fall (>0.5 m).
    assert max_pelvis_drift < 0.35, f"pelvis height drifted too far from init: {max_pelvis_drift:.3f} m"
    return True


def test_alex_wbc_standing_hold():
    result = run_simulation_app_function(_test_alex_wbc_standing_hold, headless=HEADLESS)
    assert result
