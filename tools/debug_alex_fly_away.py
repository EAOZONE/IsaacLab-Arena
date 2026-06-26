# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Headless repro for the Alex teleop "fly away" without any XR input.

Builds alex_open_microwave with the given embodiment, then steps with wrist
targets held at the current EEF poses. If the robot still explodes, the cause
is the URDF / physics setup, not OpenXR retargeting.

Run inside the container:

    /isaac-sim/python.sh tools/debug_alex_fly_away.py --embodiment alex_v2_ability_hands
    /isaac-sim/python.sh tools/debug_alex_fly_away.py --embodiment alex_v2_wbc_ability_hands
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--disable_self_collisions", action="store_true")
parser.add_argument(
    "--step_response",
    action="store_true",
    help="After settling, offset the right-wrist target by 15 cm and report tracking error over time.",
)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import warp as wp

from isaaclab_arena.embodiments.alex.alex import stabilize_alex_ability_hand_teleop_action


def main() -> None:
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_cfg = arena_builder.compose_manager_cfg()
    if args_cli.disable_self_collisions:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        print("[repro] self-collisions DISABLED")
    env = arena_builder.make_registered(env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    action_dim = env.action_space.shape[-1]
    print(f"[repro] action_dim={action_dim}")
    print(f"[repro] bodies={robot.body_names}")

    pelvis_idx = robot.body_names.index("PELVIS_LINK")
    torso_idx = robot.body_names.index("TORSO_LINK")

    for step in range(args_cli.steps):
        action = torch.zeros(action_dim, device=env.unwrapped.device)
        # Replace wrist targets with current EEF poses ("hold still" teleop input).
        action = stabilize_alex_ability_hand_teleop_action(env.unwrapped, action, force_hold_wrists=True)
        actions = action.repeat(env.unwrapped.num_envs, 1)
        with torch.inference_mode():
            env.step(actions)

        body_pos = wp.to_torch(robot.data.body_pos_w)[0]
        body_vel = wp.to_torch(robot.data.body_vel_w)[0] if hasattr(robot.data, "body_vel_w") else None
        joint_vel = wp.to_torch(robot.data.joint_vel)[0]
        max_pos = body_pos.abs().max().item()
        max_jvel = joint_vel.abs().max().item()
        nan = not torch.isfinite(body_pos).all()
        if step % 10 == 0 or nan or max_pos > 5.0:
            pelvis = body_pos[pelvis_idx].tolist()
            torso = body_pos[torso_idx].tolist()
            print(
                f"[repro] step={step:4d} pelvis={[round(v, 3) for v in pelvis]}"
                f" torso={[round(v, 3) for v in torso]}"
                f" max|body_pos|={max_pos:.3f} max|joint_vel|={max_jvel:.3f} nan={nan}"
            )
            if body_vel is not None:
                print(f"[repro]            max|body_vel|={body_vel.abs().max().item():.3f}")
            jvel_topk = torch.topk(joint_vel.abs(), k=min(6, joint_vel.numel()))
            names = [robot.joint_names[i] for i in jvel_topk.indices.tolist()]
            speeds = [round(v, 1) for v in jvel_topk.values.tolist()]
            print(f"[repro]            top joint speeds: {list(zip(names, speeds))}")
        if nan or max_pos > 100.0:
            jvel_topk = torch.topk(joint_vel.abs(), k=min(8, joint_vel.numel()))
            names = [robot.joint_names[i] for i in jvel_topk.indices.tolist()]
            print(f"[repro] EXPLODED at step {step}: top joint speeds {list(zip(names, jvel_topk.values.tolist()))}")
            break
    else:
        print("[repro] completed without explosion")

    if args_cli.step_response:
        _run_step_response(env, robot)

    env.close()


def _run_step_response(env, robot) -> None:
    """Offset the right-wrist target and report how the tracking error closes."""
    device = env.unwrapped.device
    action_dim = env.action_space.shape[-1]
    right_idx = int(robot.find_bodies(["RIGHT_GRIPPER_Z_LINK"])[0][0])

    # Hold action at the settled pose, then displace the right wrist target.
    action = torch.zeros(action_dim, device=device)
    action = stabilize_alex_ability_hand_teleop_action(env.unwrapped, action, force_hold_wrists=True)
    target_pos = action[7:10].clone() + torch.tensor([0.10, -0.05, 0.10], device=device)
    target_quat_xyzw = action[10:14].clone()
    action[7:10] = target_pos

    print(f"[step] commanding right wrist from {action[7:10] - torch.tensor([0.10, -0.05, 0.10], device=device)}")
    print(f"[step] target_pos={[round(v, 3) for v in target_pos.tolist()]}")
    for step in range(150):
        actions = action.repeat(env.unwrapped.num_envs, 1)
        with torch.inference_mode():
            env.step(actions)
        cur_pos = wp.to_torch(robot.data.body_pos_w)[0, right_idx]
        cur_quat_xyzw = wp.to_torch(robot.data.body_quat_w)[0, right_idx]
        pos_err = torch.linalg.norm(cur_pos - target_pos).item()
        quat_dot = torch.abs(torch.sum(cur_quat_xyzw * target_quat_xyzw)).clamp(max=1.0)
        rot_err = (2.0 * torch.acos(quat_dot)).item()
        if step % 5 == 0 or step == 149:
            print(f"[step] step={step:4d} pos_err={pos_err * 100:6.2f} cm rot_err={rot_err:6.3f} rad")


if __name__ == "__main__":
    main()
