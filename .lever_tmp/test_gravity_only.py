"""Does LEVER_AGAIN.usd's handle move under gravity alone (no robot contact)? If handle_angle
stays bit-exact zero for hundreds of steps despite stiffness=2000 (softened drive) and gravity
pulling on an off-axis mass, the pose *read* is almost certainly stale (Fabric/USD sync), not the
physics itself.
"""
import sys

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
args_cli = parser.parse_args(
    [
        "--headless",
        "alex_empty",
        "--embodiment",
        "alex_v2_ability_hands",
        "--usd",
        "isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd",
    ]
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import warp as wp

from isaaclab_arena.embodiments.alex.alex import build_ability_hand_joint_action
from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

arena_builder = get_arena_builder_from_cli(args_cli)
env_name, env_cfg = arena_builder.build_registered()
if hasattr(env_cfg.terminations, "success"):
    env_cfg.terminations.success = None
env_cfg.terminations.time_out = None
env = gym.make(env_name, cfg=env_cfg).unwrapped
reapply_viewer_cfg(env)
env.reset()


def object_pose():
    import omni.usd
    from pxr import Usd, UsdGeom
    from isaaclab_arena_environments.lever_scene_builder import LEVER_HANDLE_RIGID_BODY_SUFFIX

    stage = omni.usd.get_context().get_stage()
    suffix = f"/lever_again{LEVER_HANDLE_RIGID_BODY_SUFFIX}"
    candidates = [p for p in stage.Traverse() if str(p.GetPath()).endswith(suffix)]
    assert len(candidates) == 1
    tf = UsdGeom.Xformable(candidates[0]).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = tf.ExtractTranslation()
    r = tf.ExtractRotationQuat()
    imag = r.GetImaginary()
    return tuple(t), (imag[0], imag[1], imag[2], r.GetReal())


home_pos, home_quat = None, None
gripper_ids, _ = env.scene["robot"].find_bodies(["RIGHT_GRIPPER_Z_LINK", "LEFT_GRIPPER_Z_LINK"])

with torch.inference_mode():
    pos0, quat0 = object_pose()
    print(f"[GRAV] step=0 pos={pos0} quat={quat0}")
    # hold the arm exactly where it starts (34-dim action = current wrist poses + open hand)
    robot = env.scene["robot"]
    left_pos = wp.to_torch(robot.data.body_pos_w)[0, gripper_ids[1]] - env.scene.env_origins[0]
    left_quat = wp.to_torch(robot.data.body_quat_w)[0, gripper_ids[1]]
    right_pos = wp.to_torch(robot.data.body_pos_w)[0, gripper_ids[0]] - env.scene.env_origins[0]
    right_quat = wp.to_torch(robot.data.body_quat_w)[0, gripper_ids[0]]
    open_hand = build_ability_hand_joint_action(0.0, 0.0, device=env.device)
    action = torch.cat([left_pos, left_quat, right_pos, right_quat, open_hand]).unsqueeze(0)

    for i in range(300):
        env.step(action)
        if i % 30 == 0 or i == 299:
            pos, quat = object_pose()
            dpos = tuple(a - b for a, b in zip(pos, pos0))
            print(f"[GRAV] step={i+1} pos={pos} dpos={dpos} quat={quat}")

env.close()
simulation_app.close()
