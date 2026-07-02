# Temporary probe: verify lever-3 handle sits at the episode-8 grasp point.
from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless", "--enable_cameras"])
app = AppLauncher(args).app

import gymnasium as gym
import json
import numpy as np
import torch
from pathlib import Path

import pandas as pd
import warp as wp

import isaaclab_arena_environments.alex_lever_teleop_environment  # noqa: F401
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

p = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(p)
a = p.parse_args(["alex_lever_teleop", "--embodiment", "alex_v2_lever_fingers_joint_pos", "--seed", "0"])
a.enable_cameras = True
builder = get_arena_builder_from_cli(a)
name, cfg = builder.build_registered()
cfg.recorders = {}
cfg.terminations = {}

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

cfg.scene.probe_cam = CameraCfg(
    prim_path="/World/probe_cam",
    update_period=0,
    height=240,
    width=320,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
)

env = gym.make(name, cfg=cfg).unwrapped
robot = env.scene["robot"]

root = Path("datasets/alex_lever")
with open(root / "meta/info.json") as f:
    info = json.load(f)
motors = info["features"]["observation.state"]["names"]["motors"]
frames = pd.concat([pd.read_parquet(pq) for pq in sorted((root / "data").glob("*/*.parquet"))])
frame_idx = 75  # 2.5 s @ 30 Hz, grasp moment
frame = np.array(
    frames[(frames.episode_index == 8) & (frames.frame_index == frame_idx)].iloc[0]["observation.state"]
)
sim_names = [n.upper() if "ability_hand" not in n else n for n in motors]
joint_ids, _ = robot.find_joints(sim_names, preserve_order=True)
joint_ids = torch.tensor(joint_ids, dtype=torch.int32, device=env.device)
positions = torch.as_tensor(frame, device=env.device, dtype=torch.float32).unsqueeze(0)

env.reset()
robot.write_joint_position_to_sim_index(position=positions, joint_ids=joint_ids)
robot.write_data_to_sim()
env.scene.update(dt=env.sim.get_physics_dt())
for _ in range(3):
    env.sim.step(render=True)
env.scene.update(dt=env.sim.get_physics_dt())

eef_ids, _ = robot.find_bodies(["RIGHT_GRIPPER_Z_LINK"])
eef = wp.to_torch(robot.data.body_pos_w)[0, eef_ids[0]].tolist()
print("PROBE eef world", eef, flush=True)

import omni.usd
from pxr import Gf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
xf = UsdGeom.XformCache(Usd.TimeCode.Default())
handle = stage.GetPrimAtPath(
    "/World/envs/env_0/lever_layout/geometry/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/Handle_1"
)
pts_min = None
pts_max = None
for child in Usd.PrimRange(handle):
    if not child.IsA(UsdGeom.Mesh):
        continue
    ext = UsdGeom.Mesh(child).GetExtentAttr().Get()
    m = xf.GetLocalToWorldTransform(child)
    for i in range(8):
        corner = Gf.Vec3d(ext[(i >> 0) & 1][0], ext[(i >> 1) & 1][1], ext[(i >> 2) & 1][2])
        w = m.Transform(corner)
        if pts_min is None:
            pts_min = [w[0], w[1], w[2]]
            pts_max = [w[0], w[1], w[2]]
        else:
            pts_min = [min(pts_min[k], w[k]) for k in range(3)]
            pts_max = [max(pts_max[k], w[k]) for k in range(3)]
center = [(pts_min[k] + pts_max[k]) / 2.0 for k in range(3)]
target = [0.16465, 0.01114, 0.87947]
print("PROBE handle center", center, flush=True)
print("PROBE handle-target delta", [center[k] - target[k] for k in range(3)], flush=True)
print("PROBE eef-handle delta", [eef[k] - center[k] for k in range(3)], flush=True)

cam = env.scene.sensors["probe_cam"]
positions_cam = torch.tensor([[1.6, -1.6, 1.5]], device=env.device)
targets_cam = torch.tensor([[0.1, 0.0, 0.85]], device=env.device)
cam.set_world_poses_from_view(positions_cam, targets_cam)
for _ in range(10):
    env.sim.step(render=True)
cam.update(dt=0.0)
rgb = cam.data.output["rgb"]
if not isinstance(rgb, torch.Tensor):
    rgb = wp.to_torch(rgb)
img = rgb[0].detach().cpu().numpy().astype(np.uint8)
from PIL import Image

Image.fromarray(img[..., :3]).save("/workspaces/isaaclab_arena/_probe_grasp.png")
print("PROBE saved image", flush=True)
app.close()
