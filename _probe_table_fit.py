# Temporary: board vs table placement sanity check.
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

p = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(p)
a = p.parse_args(["alex_lever_teleop", "--embodiment", "alex_v2_lever_fingers_joint_pos", "--seed", "0"])
builder = get_arena_builder_from_cli(a)
name, cfg = builder.build_registered()
cfg.recorders = {}
cfg.terminations = {}
env = gym.make(name, cfg=cfg).unwrapped
env.reset()
for _ in range(2):
    env.sim.step(render=False)

import omni.usd
from pxr import Gf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
xf = UsdGeom.XformCache(Usd.TimeCode.Default())
layout = stage.GetPrimAtPath("/World/envs/env_0/lever_layout")
root_pos = xf.GetLocalToWorldTransform(layout).ExtractTranslation()
b_min = b_max = None
for child in Usd.PrimRange(layout):
    if not child.IsA(UsdGeom.Mesh):
        continue
    ext = UsdGeom.Mesh(child).GetExtentAttr().Get()
    if ext is None:
        continue
    m = xf.GetLocalToWorldTransform(child)
    for i in range(8):
        corner = Gf.Vec3d(ext[(i >> 0) & 1][0], ext[(i >> 1) & 1][1], ext[(i >> 2) & 1][2])
        w = m.Transform(corner)
        if b_min is None:
            b_min = [w[0], w[1], w[2]]
            b_max = [w[0], w[1], w[2]]
        else:
            b_min = [min(b_min[k], w[k]) for k in range(3)]
            b_max = [max(b_max[k], w[k]) for k in range(3)]

table = stage.GetPrimAtPath("/World/envs/env_0/table")
tpos = xf.GetLocalToWorldTransform(table).ExtractTranslation()
table_top_z = tpos[2] + 0.02
board_cx = (b_min[0] + b_max[0]) / 2.0
board_cy = (b_min[1] + b_max[1]) / 2.0
print("layout root", tuple(root_pos), flush=True)
print("board bounds min", b_min, "max", b_max, flush=True)
print("board center xy", board_cx, board_cy, flush=True)
print("table center", tuple(tpos), flush=True)
print("table top z", table_top_z, flush=True)
print("gap board_bottom - table_top", b_min[2] - table_top_z, flush=True)
print("table xy offset from board center", tpos[0] - board_cx, tpos[1] - board_cy, flush=True)
app.close()
