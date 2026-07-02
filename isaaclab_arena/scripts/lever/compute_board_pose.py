# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Print lever-board prim world positions and solve layout root for a target point."""

from isaaclab.app import AppLauncher

import argparse
import math

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
from pathlib import Path
from pxr import Gf, Usd, UsdGeom
import omni.usd

import isaaclab_arena_environments.alex_lever_teleop_environment  # noqa: F401
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

TARGET = np.array([0.08387, -0.04905, 0.97803 + 0.03], dtype=np.float64)

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
env.reset()
for _ in range(2):
    env.sim.step(render=False)

stage = omni.usd.get_context().get_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
layout_root = stage.GetPrimAtPath("/World/envs/env_0/lever_layout")
assert layout_root.IsValid(), "lever_layout not found"

xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
layout_world = xf_cache.GetLocalToWorldTransform(layout_root)
layout_origin = np.array(layout_world.ExtractTranslation(), dtype=np.float64)

out_lines = []
def log(msg: str) -> None:
    out_lines.append(msg)
    print(msg, flush=True)

log(f"layout_root world origin {layout_origin.tolist()}")

candidates = []
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "/lever_layout/" not in path:
        continue
    leaf = path.split("/")[-1]
    if not any(k in leaf for k in ("Lever", "lever", "Valve", "Joystick", "Handle", "Cap")):
        continue
    if not prim.IsA(UsdGeom.Xformable):
        continue
    # ComputeAlignedRange applies the bound's world matrix; GetBox() would return
    # the prim-local range and silently drop the spawn transform.
    c = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
    if abs(c[0]) + abs(c[1]) + abs(c[2]) < 1e-6:
        continue
    local = layout_world.GetInverse().Transform(Gf.Vec3d(float(c[0]), float(c[1]), float(c[2])))
    local_np = np.array([local[0], local[1], local[2]], dtype=np.float64)
    world_np = np.array([c[0], c[1], c[2]], dtype=np.float64)
    candidates.append((path, world_np, local_np))

candidates.sort(key=lambda item: item[1][0])
log("\nLever-like prims (world, local-to-layout-root):")
for path, world, local in candidates:
    log(f"  {path.split('/')[-2]}/{path.split('/')[-1]}")
    log(f"    world {world.tolist()}")
    log(f"    local {local.tolist()}")

# Pick nearest to target in world (current placement)
best = min(candidates, key=lambda item: np.linalg.norm(item[1] - TARGET))
log(f"\nNearest to target {TARGET.tolist()} (current placement):")
log(f"  {best[0]}")
log(f"  world {best[1].tolist()} delta {(TARGET - best[1]).tolist()}")

shift = TARGET - best[1]
new_layout = layout_origin + shift
log(f"\nSuggested layout root shift (add to current origin): {shift.tolist()}")
log(f"Suggested layout root world xyz: {new_layout.tolist()}")

# Table: current table center from stage
table_prim = stage.GetPrimAtPath("/World/envs/env_0/table")
table_world = xf_cache.GetLocalToWorldTransform(table_prim)
table_origin = np.array(table_world.ExtractTranslation(), dtype=np.float64)
table_shift = new_layout - layout_origin
new_table = table_origin + table_shift
log(f"Current table origin {table_origin.tolist()}")
log(f"Suggested table origin (shift with board): {new_table.tolist()}")

Path("/tmp/board_pose_result.txt").write_text("\n".join(out_lines) + "\n")

env.close()
app.close()
