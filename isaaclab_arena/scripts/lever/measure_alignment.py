# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Headless: right EEF vs lever-board object positions at a dataset frame."""

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym
import json
import numpy as np
import pandas as pd
import torch
import warp as wp
from pathlib import Path

import isaaclab_arena_environments.alex_lever_teleop_environment  # noqa: F401
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli
from pxr import Usd, UsdGeom
import omni.usd

parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
args_cli = parser.parse_args(
    ["alex_lever_teleop", "--embodiment", "alex_v2_lever_fingers_joint_pos", "--seed", "0"]
)

arena_builder = get_arena_builder_from_cli(args_cli)
env_name, env_cfg = arena_builder.build_registered()
env_cfg.recorders = {}
env_cfg.terminations = {}
env = gym.make(env_name, cfg=env_cfg).unwrapped
robot = env.scene["robot"]

root = Path("datasets/alex_lever")
with open(root / "meta/info.json") as f:
    info = json.load(f)
motors = info["features"]["observation.state"]["names"]["motors"]
frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))])
frame_idx = 75  # 2.5s @ 30 Hz
frame = np.array(frames[(frames.episode_index == 8) & (frames.frame_index == frame_idx)].iloc[0]["observation.state"])
sim_names = [n.upper() if "ability_hand" not in n else n for n in motors]
joint_ids, _ = robot.find_joints(sim_names, preserve_order=True)
joint_ids = torch.tensor(joint_ids, dtype=torch.int32, device=env.device)
positions = torch.as_tensor(frame, device=env.device, dtype=torch.float32).unsqueeze(0)

env.reset()
robot.write_joint_position_to_sim_index(position=positions, joint_ids=joint_ids)
robot.write_data_to_sim()
env.scene.update(dt=env.sim.get_physics_dt())
for _ in range(3):
    env.sim.step(render=False)

eef_ids, _ = robot.find_bodies(["RIGHT_GRIPPER_Z_LINK"])
eef = wp.to_torch(robot.data.body_pos_w)[0, eef_ids[0]].tolist()
print(f"RIGHT_GRIPPER_Z_LINK {eef}")

stage = omni.usd.get_context().get_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
objects = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "/lever_layout/" not in path:
        continue
    name = path.split("/")[-1]
    if not prim.IsA(UsdGeom.Xformable):
        continue
    if name in {"Layout_v9", "lever_layout", "geometry"}:
        continue
    # ComputeAlignedRange applies the bound's world matrix; GetBox() would return
    # the prim-local range and silently drop the spawn transform.
    c = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
    if abs(c[0]) + abs(c[1]) + abs(c[2]) < 1e-6:
        continue
    objects.append((path, (c[0], c[1], c[2])))

# nearest object to EEF in XY
eef_t = np.array(eef)
objects.sort(key=lambda item: np.linalg.norm(np.array(item[1]) - eef_t))
print("Nearest to EEF:")
for path, pos in objects[:8]:
    d = np.linalg.norm(np.array(pos) - eef_t)
    print(f"  d={d:.3f} {path.split('/')[-2]}/{path.split('/')[-1]} {pos}")

env.close()
app.close()
