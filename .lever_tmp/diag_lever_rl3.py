# Confirmatory diagnostic: send a VALID "hold current pose" action (live wrist
# poses with proper unit quaternions, [left_pos(3), left_quat(4), right_pos(3),
# right_quat(4), hand(...)]) instead of all-zeros, and check whether:
#  (a) "IK contains NaN" failures disappear
#  (b) the arm stops sagging under gravity
#  (c) hinge_turn_progress stays near 0 (no spurious handle contact)

import argparse
import torch
import traceback

from isaaclab.app import AppLauncher

_launcher_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_launcher_parser)
_launcher_args, _ = _launcher_parser.parse_known_args(["--headless"])
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

        robot = env.scene["robot"]
        lever = env.scene["lever_revolute"]
        env_origins = env.scene.env_origins
        right_gripper_idx = robot.find_bodies("RIGHT_GRIPPER_Z_LINK")[0][0]
        left_gripper_idx = robot.find_bodies("LEFT_GRIPPER_Z_LINK")[0][0]

        action_dim = env.action_space.shape[-1]
        hand_dim = action_dim - 14
        print("action_dim:", action_dim, "hand_dim:", hand_dim)

        with torch.inference_mode():
            for step in range(60):
                # Build a "hold current pose" action fresh each step from the LIVE body pose,
                # per-env (batched), mirroring record_scripted_lever_demos.py's _wrist_pose pattern.
                left_pos = wp.to_torch(robot.data.body_pos_w)[:, left_gripper_idx] - env_origins
                left_quat = wp.to_torch(robot.data.body_quat_w)[:, left_gripper_idx]
                right_pos = wp.to_torch(robot.data.body_pos_w)[:, right_gripper_idx] - env_origins
                right_quat = wp.to_torch(robot.data.body_quat_w)[:, right_gripper_idx]
                hand = torch.zeros(env.num_envs, hand_dim, device=env.device)
                actions = torch.cat([left_pos, left_quat, right_pos, right_quat, hand], dim=1)

                obs, rew, terminated, truncated, info = env.step(actions)
                rm = env.reward_manager
                if step % 10 == 0 or step == 59:
                    lever_pos = wp.to_torch(lever.data.root_pos_w)[0] - env_origins[0]
                    hand_pos = wp.to_torch(robot.data.body_pos_w)[0, right_gripper_idx] - env_origins[0]
                    dist = torch.norm(hand_pos - lever_pos).item()
                    term_str = " ".join(f"{n}={rm._episode_sums[n][0].item():.4f}" for n in rm.active_terms)
                    print(
                        f"step={step:3d} hand_pos_local={hand_pos.tolist()} hand_lever_dist={dist:.4f} "
                        f"reward={rew[0].item():.5f} terminated={terminated[0].item()} {term_str}"
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
