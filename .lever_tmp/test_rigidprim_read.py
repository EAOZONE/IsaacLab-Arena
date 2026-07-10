"""Does isaacsim.core.prims.RigidPrim give a LIVE (Fabric-synced) pose for Handle_1, unlike raw
UsdGeom.Xformable.ComputeLocalToWorldTransform (proven stale via test_gravity_only.py)?
"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

import sys

parser2 = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser2)
args_cli2 = parser2.parse_args(
    [
        "--headless",
        "alex_empty",
        "--embodiment",
        "alex_v2_ability_hands",
        "--usd",
        "isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd",
    ]
)

import torch
import gymnasium as gym
import warp as wp

from isaaclab_arena.embodiments.alex.alex import build_ability_hand_joint_action
from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

arena_builder = get_arena_builder_from_cli(args_cli2)
env_name, env_cfg = arena_builder.build_registered()
if hasattr(env_cfg.terminations, "success"):
    env_cfg.terminations.success = None
env_cfg.terminations.time_out = None
env = gym.make(env_name, cfg=env_cfg).unwrapped
reapply_viewer_cfg(env)
env.reset()

from isaacsim.core.prims import RigidPrim

handle_prim = RigidPrim(
    "/World/envs/env_0/lever_again/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/"
    "base_link_1/base_link/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/Handle_1"
)

gripper_ids, _ = env.scene["robot"].find_bodies(["RIGHT_GRIPPER_Z_LINK", "LEFT_GRIPPER_Z_LINK"])
robot = env.scene["robot"]

with torch.inference_mode():
    pos0, quat0 = handle_prim.get_world_poses()
    print(f"[RIGIDPRIM] step=0 pos={pos0} quat={quat0}")

    left_pos = wp.to_torch(robot.data.body_pos_w)[0, gripper_ids[1]] - env.scene.env_origins[0]
    left_quat = wp.to_torch(robot.data.body_quat_w)[0, gripper_ids[1]]
    right_pos = wp.to_torch(robot.data.body_pos_w)[0, gripper_ids[0]] - env.scene.env_origins[0]
    right_quat = wp.to_torch(robot.data.body_quat_w)[0, gripper_ids[0]]
    open_hand = build_ability_hand_joint_action(0.0, 0.0, device=env.device)
    action = torch.cat([left_pos, left_quat, right_pos, right_quat, open_hand]).unsqueeze(0)

    for i in range(200):
        env.step(action)
        if i % 20 == 0 or i == 199:
            pos, quat = handle_prim.get_world_poses()
            print(f"[RIGIDPRIM] step={i+1} pos={pos} quat={quat}")

env.close()
simulation_app.close()
