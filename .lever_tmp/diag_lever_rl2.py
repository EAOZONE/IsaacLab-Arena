# Diagnostic: compare Pink IK solver stability under zero actions for
# use_teleop_actuators=True (stiff) vs False (weak "RL" actuators).
# Usage: python diag_lever_rl2.py --teleop_actuators

import argparse
import math
import sys
import torch
import traceback

from isaaclab.app import AppLauncher

_launcher_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_launcher_parser)
_launcher_args, _ = _launcher_parser.parse_known_args(["--headless"])
_app_launcher = AppLauncher(_launcher_args)
simulation_app = _app_launcher.app

_diag_parser = argparse.ArgumentParser()
_diag_parser.add_argument("--teleop_actuators", action="store_true")
_diag_args = _diag_parser.parse_args(
    [a for a in sys.argv[1:] if a not in ("--headless",)]
)

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: E402
from isaaclab_arena.scene.scene import Scene  # noqa: E402
from isaaclab_arena.tasks.lever_turn_task import LeverTurnTaskRL  # noqa: E402
from isaaclab_arena.utils.pose import Pose  # noqa: E402
from isaaclab_arena.assets.registries import AssetRegistry  # noqa: E402
from isaaclab_arena_environments import lever_scene_builder  # noqa: E402

import gymnasium as gym  # noqa: E402


def main() -> bool:
    asset_registry = AssetRegistry()
    ground_plane = asset_registry.get_asset_by_name("ground_plane")()
    light = asset_registry.get_asset_by_name("light")()
    ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

    lever_assets, lever_object = lever_scene_builder.build_lever_scene_assets(
        usd_path="isaaclab_arena/assets/lever_sim/Lever_revolute.usd",
        usd_pos=lever_scene_builder.LEVER_USD_DEFAULT_POS,
        usd_yaw=lever_scene_builder.LEVER_USD_DEFAULT_YAW,
        usd_scale=lever_scene_builder.LEVER_USD_DEFAULT_SCALE,
        lever_dr=False,
        table="seattle_lab",
    )

    embodiment = asset_registry.get_asset_by_name("alex_v2_ability_hands")(
        concatenate_observation_terms=True,
        use_teleop_actuators=_diag_args.teleop_actuators,
    )
    spawn_pos = (-0.4, -0.48682, 0.94296)
    embodiment.set_initial_pose(Pose(position_xyz=spawn_pos, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    scene = Scene(assets=[ground_plane, light, *lever_assets])
    task = LeverTurnTaskRL(lever_object=lever_object, embodiment=embodiment)

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="alex_lever_turn_diag",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    parser = get_isaaclab_arena_cli_parser()
    args_cli = parser.parse_args(["--num_envs", "4"])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    try:
        import warp as wp

        robot = env.scene["robot"]
        lever = env.scene["lever_revolute"]
        env_origins = env.scene.env_origins
        right_gripper_idx = robot.find_bodies("RIGHT_GRIPPER_Z_LINK")[0][0]

        with torch.inference_mode():
            for step in range(60):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                obs, rew, terminated, truncated, info = env.step(actions)
                if step % 10 == 0 or step == 59:
                    lever_pos = wp.to_torch(lever.data.root_pos_w)[0] - env_origins[0]
                    hand_pos = wp.to_torch(robot.data.body_pos_w)[0, right_gripper_idx] - env_origins[0]
                    dist = torch.norm(hand_pos - lever_pos).item()
                    print(
                        f"[teleop_actuators={_diag_args.teleop_actuators}] step={step:3d} "
                        f"hand_pos_local={hand_pos.tolist()} hand_lever_dist={dist:.4f}"
                    )
    finally:
        env.close()

    return True


if __name__ == "__main__":
    ok = False
    try:
        ok = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    print("RESULT_OK" if ok else "RESULT_FAIL")
