"""Show how sim spawn rotation reorients the whole lever board."""

import math

import torch
from isaaclab.utils.math import quat_from_euler_xyz
from pxr import Gf, Usd, UsdGeom

path = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
stage = Usd.Stage.Open(path)

BASE = (
    "/World/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
)
HANDLE = (
    BASE
    + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"
)

cache = UsdGeom.XformCache()

lever_yaw_rad = math.radians(90.0 + 90.0)
spawn_quat = quat_from_euler_xyz(
    torch.tensor([math.pi / 2.0]),
    torch.tensor([0.0]),
    torch.tensor([lever_yaw_rad]),
)[0].tolist()
spawn_rot = Gf.Rotation(Gf.Quatd(spawn_quat[3], Gf.Vec3d(spawn_quat[0], spawn_quat[1], spawn_quat[2])))


def describe(label: str, xf: Gf.Matrix4d, spawn: Gf.Rotation | None = None) -> None:
    rot = xf.ExtractRotation()
    if spawn is not None:
        rot = spawn * rot
    for axis_name, axis in [("+X", Gf.Vec3d(1, 0, 0)), ("+Y", Gf.Vec3d(0, 1, 0)), ("+Z", Gf.Vec3d(0, 0, 1))]:
        d = rot.TransformDir(axis)
        d.Normalize()
        print(f"  {label} local {axis_name} -> world ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})")


print("=== Native USD (Y-up stage, no spawn transform) ===")
base_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
handle_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))
describe("base", base_xf)
describe("handle", handle_xf)

# vector from base to handle (approx lever stick direction in native frame)
offset = handle_xf.ExtractTranslation() - base_xf.ExtractTranslation()
offset.Normalize()
print(f"  base->handle offset dir: ({offset[0]:+.2f}, {offset[1]:+.2f}, {offset[2]:+.2f})")

print("\n=== After sim spawn (Z-up, roll=90°, yaw=180°) ===")
describe("base", base_xf, spawn_rot)
describe("handle", handle_xf, spawn_rot)
spawn_offset = spawn_rot.TransformDir(offset)
spawn_offset.Normalize()
print(f"  base->handle offset dir: ({spawn_offset[0]:+.2f}, {spawn_offset[1]:+.2f}, {spawn_offset[2]:+.2f})")

# pegboard normal: Layout_v9 is probably the root layout
layout = stage.GetPrimAtPath("/World/Layout_v9")
if layout:
    layout_xf = cache.GetLocalToWorldTransform(layout)
    print("\n=== Layout pegboard normal (native local +Z) ===")
    describe("layout", layout_xf)
    describe("layout (after spawn)", layout_xf, spawn_rot)
