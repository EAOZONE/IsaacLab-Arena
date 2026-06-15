# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build-and-step smoke test for the bare ``alex_playground`` environment."""

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_alex_playground_builds_and_steps(simulation_app):
    import gymnasium as gym
    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.alex_playground_environment import AlexPlaygroundEnvironment

    args = get_isaaclab_arena_cli_parser().parse_args([])
    args.embodiment = "alex_v2_ability_hands"
    args.background = "packing_table"
    args.teleop_device = None
    args.enable_cameras = False

    arena_env = AlexPlaygroundEnvironment().get_env(args)
    assert arena_env.task is None, "playground env must have no task"

    builder = ArenaEnvBuilder(arena_env, args)
    if arena_env.name in gym.registry:
        del gym.registry[arena_env.name]
    env = builder.make_registered()

    env.reset()
    for _ in range(2):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
    return True


def test_alex_playground_builds_and_steps():
    result = run_simulation_app_function(_test_alex_playground_builds_and_steps, headless=HEADLESS)
    assert result
