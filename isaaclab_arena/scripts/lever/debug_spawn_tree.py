# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Print lever_layout prim tree after scene spawn."""

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym

import isaaclab_arena_environments.alex_lever_teleop_environment  # noqa: F401
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli
from pxr import UsdPhysics

p = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(p)
a = p.parse_args(["alex_lever_teleop", "--embodiment", "alex_v2_lever_fingers_joint_pos", "--seed", "0"])
builder = get_arena_builder_from_cli(a)
name, cfg = builder.build_registered()
cfg.recorders = {}
cfg.terminations = {}

try:
    env = gym.make(name, cfg=cfg).unwrapped
except Exception as exc:
    print("ENV FAIL", exc, flush=True)

import omni.usd

stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/envs/env_0/lever_layout")
print("lever_layout valid", root.IsValid(), "children:", [c.GetName() for c in root.GetChildren()], flush=True)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "/lever_layout" not in path:
        continue
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
        print(path, prim.GetTypeName(), flush=True)

app.close()
