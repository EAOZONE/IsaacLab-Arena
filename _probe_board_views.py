# Temporary probe: render alex_lever_teleop at ep8's grasp frame for board-placement tuning.
# Passes through the env's own --board_yaw / --board_offset_x / --board_offset_y CLI args.
from isaaclab.app import AppLauncher

import argparse
import sys

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless", "--enable_cameras"])
app = AppLauncher(args).app

import gymnasium as gym
import json
import numpy as np
import pandas as pd
import torch
import warp as wp
from pathlib import Path
from PIL import Image

import isaaclab_arena_environments.alex_lever_teleop_environment  # noqa: F401
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

p = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(p)
extra = [a for a in sys.argv[1:] if not a.startswith("--tag")]
tag = "base"
if "--tag" in sys.argv:
    tag = sys.argv[sys.argv.index("--tag") + 1]
    extra = [a for a in sys.argv[1:] if a != "--tag" and a != tag]
a = p.parse_args(
    ["alex_lever_teleop", "--embodiment", "alex_v2_lever_fingers_joint_pos", "--seed", "0"] + extra
)
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
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
)

env = gym.make(name, cfg=cfg).unwrapped
robot = env.scene["robot"]

# Pose the robot at ep8's closest-approach frame.
root = Path("datasets/alex_lever")
with open(root / "meta/info.json") as f:
    info = json.load(f)
motors = info["features"]["observation.state"]["names"]["motors"]
frames = pd.concat([pd.read_parquet(pq) for pq in sorted((root / "data").glob("*/*.parquet"))])
GRASP_FRAME = 156
ep = frames[frames.episode_index == 8].sort_values("frame_index")
sim_names = [n.upper() if "ability_hand" not in n else n for n in motors]
joint_ids, _ = robot.find_joints(sim_names, preserve_order=True)
joint_ids = torch.tensor(joint_ids, dtype=torch.int32, device=env.device)
zero_vel = torch.zeros((1, len(sim_names)), device=env.device)

env.reset()
# Progressive replay up to the grasp frame, mirroring playback_lerobot_dataset.py.
for _, row in ep.iterrows():
    fi = int(row["frame_index"])
    if fi > GRASP_FRAME:
        break
    positions = torch.as_tensor(
        np.array(row["observation.state"]), device=env.device, dtype=torch.float32
    ).unsqueeze(0)
    robot.write_joint_position_to_sim_index(position=positions, joint_ids=joint_ids)
    robot.write_joint_velocity_to_sim_index(velocity=zero_vel, joint_ids=joint_ids)
    robot.set_joint_position_target_index(target=positions, joint_ids=joint_ids)
    robot.write_data_to_sim()
    env.sim.step(render=False)
env.scene.update(dt=env.sim.get_physics_dt())

# Fingertip vs blue-valve handle.
from pxr import Usd, UsdGeom
import omni.usd

stage = omni.usd.get_context().get_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
handle = None
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "/lever_layout/" in path and path.endswith("Handle_1/Handle"):
        handle = np.array(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint())
        break
assert handle is not None
tip_ids, tip_names = robot.find_bodies(["right_ability_hand_index_L2"])
tip = wp.to_torch(robot.data.body_pos_w)[0, tip_ids[0]].cpu().numpy()
print(f"PROBE[{tag}] handle {handle.tolist()}", flush=True)
print(f"PROBE[{tag}] right index tip {tip.tolist()}", flush=True)
print(f"PROBE[{tag}] tip-to-handle delta {(handle - tip).tolist()} dist {np.linalg.norm(handle - tip):.3f}", flush=True)

cam = env.scene.sensors["probe_cam"]
views = {
    "side": ([1.6, -1.6, 1.5], [0.15, 0.0, 0.85]),
    "top": ([0.15, 0.0, 2.6], [0.15, 0.01, 0.85]),
    "front": ([1.8, 0.0, 1.1], [0.0, 0.0, 0.9]),
    "hand": ([0.6, -0.5, 1.1], [0.16, 0.01, 0.88]),
}
for vname, (pos, look) in views.items():
    cam.set_world_poses_from_view(
        torch.tensor([pos], device=env.device), torch.tensor([look], device=env.device)
    )
    for _ in range(8):
        env.sim.step(render=True)
    env.scene.update(dt=env.sim.get_physics_dt())
    cam.update(dt=0.0)
    rgb = cam.data.output["rgb"]
    arr = rgb if isinstance(rgb, torch.Tensor) else wp.to_torch(rgb)
    img = arr[0].detach().cpu().numpy().astype(np.uint8)
    out = f"/workspaces/isaaclab_arena/_probe_{tag}_{vname}.png"
    Image.fromarray(img[..., :3]).save(out)
    print(f"PROBE[{tag}] saved {out}", flush=True)

app.close()
