# Temporary probe: render a side view of alex_lever_teleop to locate the board.
from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless", "--enable_cameras"])
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

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
    spawn=sim_utils.PinholeCameraCfg(focal_length=14.0),
)

env = gym.make(name, cfg=cfg).unwrapped
env.reset()



def to_numpy(rgb):
    if isinstance(rgb, torch.Tensor):
        arr = rgb
    else:
        import warp as wp

        arr = wp.to_torch(rgb)
    return arr[0].detach().cpu().numpy().astype(np.uint8)

cam = env.scene.sensors["probe_cam"]
positions = torch.tensor([[2.0, -2.0, 1.5]], device=env.device)
targets = torch.tensor([[0.0, 0.0, 0.8]], device=env.device)
cam.set_world_poses_from_view(positions, targets)

for _ in range(30):
    env.sim.step(render=True)
env.scene.update(dt=env.sim.get_physics_dt())
cam.update(dt=0.0)
img = to_numpy(cam.data.output["rgb"])
from PIL import Image

Image.fromarray(img[..., :3]).save("/workspaces/isaaclab_arena/_probe_scene.png")
print("PROBE saved image", img.shape, flush=True)

app.close()
