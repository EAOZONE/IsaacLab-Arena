# Temporary probe: true world center of the lever-3 handle meshes vs layout root.
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

root = stage.GetPrimAtPath("/World/envs/env_0/lever_layout")
root_pos = xf.GetLocalToWorldTransform(root).ExtractTranslation()
print("PROBE root world", tuple(root_pos), flush=True)

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
        corner = Gf.Vec3d(
            ext[(i >> 0) & 1][0],
            ext[(i >> 1) & 1][1],
            ext[(i >> 2) & 1][2],
        )
        w = m.Transform(corner)
        if pts_min is None:
            pts_min = [w[0], w[1], w[2]]
            pts_max = [w[0], w[1], w[2]]
        else:
            pts_min = [min(pts_min[k], w[k]) for k in range(3)]
            pts_max = [max(pts_max[k], w[k]) for k in range(3)]

center = [(pts_min[k] + pts_max[k]) / 2.0 for k in range(3)]
print("PROBE handle mesh world bounds min", pts_min, "max", pts_max, flush=True)
print("PROBE handle mesh world center", center, flush=True)
print("PROBE offset center-root", [center[k] - root_pos[k] for k in range(3)], flush=True)

# Board world bounds (all meshes) for table-height sanity
b_min = None
b_max = None
layout = stage.GetPrimAtPath("/World/envs/env_0/lever_layout")
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
print("PROBE board world bounds min", b_min, "max", b_max, flush=True)
print("PROBE board bounds rel root min", [b_min[k] - root_pos[k] for k in range(3)], "max", [b_max[k] - root_pos[k] for k in range(3)], flush=True)
app.close()
