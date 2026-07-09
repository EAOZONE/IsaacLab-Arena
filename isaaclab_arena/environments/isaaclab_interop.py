# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse


def _kit_is_running() -> bool:
    """Return True when Omniverse Kit is already up (``AppLauncher`` was started)."""
    import sys

    mod = sys.modules.get("omni.kit.app")
    if mod is None:
        return False
    try:
        app = mod.get_app()
        return app is not None and app.is_running()
    except Exception:
        return False


def environment_registration_callback() -> list[str]:
    """Register an Isaac Lab-Arena environment for Isaac Lab's RSL-RL ``train.py`` script.

    Passed via ``--external_callback``. ``train.py`` imports Isaac Lab modules before
    invoking this callback, so we must start ``SimulationApp`` here — **before** importing
    or building any Arena scene/embodiment code that touches ``pxr`` — then register the
    gym task. ``train.py``'s ``launch_simulation()`` sees Kit is already running and skips
    a second ``AppLauncher`` startup.

    Example::

        python submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \\
            --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \\
            --task alex_lever_turn --num_envs 64 --max_iterations 4000
    """
    from isaaclab.app import AppLauncher

    # Phase 1: parse only launcher flags and start SimulationApp before any pxr imports.
    launcher_parser = argparse.ArgumentParser()
    launcher_parser.add_argument(
        "--task", type=str, required=True, help="Name of the IsaacLab Arena environment to register."
    )
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args()
    if not _kit_is_running():
        AppLauncher(launcher_args)

    # Phase 2: safe to import Arena now that Kit is up.
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import add_isaac_lab_cli_args, add_isaaclab_arena_cli_args
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.cli import ensure_environments_registered

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Name of the IsaacLab Arena environment to register.")
    AppLauncher.add_app_launcher_args(parser)
    add_isaac_lab_cli_args(parser)
    add_isaaclab_arena_cli_args(parser)

    args, _ = parser.parse_known_args()
    ensure_environments_registered()
    environment = EnvironmentRegistry().get_component_by_name(args.task)()
    environment.add_cli_args(parser)
    args, remaining_args = parser.parse_known_args()

    isaaclab_arena_environment = environment.get_env(args)
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args)
    env_builder.build_registered()
    return remaining_args
