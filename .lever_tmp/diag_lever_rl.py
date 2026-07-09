# Diagnostic script (not a test): checks whether hinge_angle drifts under zero
# actions (settle-artifact bug) and reports reach distance / reward breakdown
# at reset for the alex_lever_turn RL task.

import argparse
import torch
import traceback

from isaaclab.app import AppLauncher

_launcher_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_launcher_parser)
_launcher_args, _ = _launcher_parser.parse_known_args(["--headless", "--enable_cameras"])
_app_launcher = AppLauncher(_launcher_args)
simulation_app = _app_launcher.app

from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder  # noqa: E402
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena_environments.alex_lever_turn_environment import AlexLeverTurnEnvironment  # noqa: E402

import gymnasium as gym  # noqa: E402


def main() -> bool:
    parser = get_isaaclab_arena_cli_parser()
    env_obj = AlexLeverTurnEnvironment()
    env_obj.add_cli_args(parser)
    args_cli = parser.parse_args(["--num_envs", "4"])

    isaaclab_arena_environment = env_obj.get_env(args_cli)
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
    name, cfg = env_builder.build_registered()
    env = gym.make(name, cfg=cfg).unwrapped
    env.reset()

    try:
        import warp as wp

        print("scene keys:", list(env.scene.keys()))
        robot = env.scene["robot"]
        lever = env.scene["lever_revolute"]
        env_origins = env.scene.env_origins
        right_gripper_idx = robot.find_bodies("RIGHT_GRIPPER_Z_LINK")[0][0]
        print("right_gripper_idx:", right_gripper_idx)

        with torch.inference_mode():
            for step in range(60):
                actions = torch.zeros(env.action_space.shape, device=env.device)
                obs, rew, terminated, truncated, info = env.step(actions)
                rm = env.reward_manager
                term_cfgs = {n: rm.get_term_cfg(n) for n in rm.active_terms}
                if step % 10 == 0 or step == 59:
                    lever_pos = wp.to_torch(lever.data.root_pos_w)[0] - env_origins[0]
                    hand_pos = wp.to_torch(robot.data.body_pos_w)[0, right_gripper_idx] - env_origins[0]
                    dist = torch.norm(hand_pos - lever_pos).item()
                    term_str = " ".join(
                        f"{n}={rm._episode_sums[n][0].item():.4f}" for n in rm.active_terms
                    )
                    print(
                        f"step={step:3d} lever_pos_local={lever_pos.tolist()} hand_pos_local={hand_pos.tolist()} "
                        f"hand_lever_dist={dist:.4f} reward={rew[0].item():.5f} terminated={terminated[0].item()} {term_str}"
                    )
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view
        import omni.kit.viewport.utility as vp_util

        lever_pos_np = wp.to_torch(lever.data.root_pos_w)[0].cpu().numpy()
        center = lever_pos_np + np.array([0.0, 0.0, 0.1])
        set_camera_view(eye=center + np.array([0.9, -0.4, 0.6]), target=center)
        for _ in range(30):
            simulation_app.update()
        vp = vp_util.get_active_viewport()
        vp_util.capture_viewport_to_file(vp, "/workspaces/isaaclab_arena/.lever_tmp/diag_reach.png")
        for _ in range(30):
            simulation_app.update()
        print("saved screenshot")

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
