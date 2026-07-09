"""Compare lever handle direction: raw USD vs sim spawn transform."""

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
handle_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))

# Approximate handle "stick" direction as local +Y in handle frame (common for valve handles)
local_y = Gf.Vec3d(0, 1, 0)
native_dir = handle_xf.TransformDir(local_y)
native_dir.Normalize()
print("Native USD (Y-up stage, q=0) handle +Y direction in stage space:")
print(f"  ({native_dir[0]:.3f}, {native_dir[1]:.3f}, {native_dir[2]:.3f})")
print(f"  angle from stage +Y (up): {math.degrees(math.acos(max(-1, min(1, native_dir[1])))):.1f} deg")

# Sim spawn rotation from lever_scene_builder (roll=pi/2, yaw=90+usd_yaw, usd_yaw=90 default)
lever_yaw_rad = math.radians(90.0 + 90.0)
spawn_quat = quat_from_euler_xyz(
    torch.tensor([math.pi / 2.0]),
    torch.tensor([0.0]),
    torch.tensor([lever_yaw_rad]),
)[0].tolist()
spawn_rot = Gf.Rotation(Gf.Quatd(spawn_quat[3], Gf.Vec3d(spawn_quat[0], spawn_quat[1], spawn_quat[2])))

sim_dir = spawn_rot.TransformDir(native_dir)
sim_dir.Normalize()
print("\nAfter sim spawn rot (Z-up world, default usd_yaw=90) handle direction:")
print(f"  ({sim_dir[0]:.3f}, {sim_dir[1]:.3f}, {sim_dir[2]:.3f})")
print(f"  angle from world +Z (up): {math.degrees(math.acos(max(-1, min(1, sim_dir[2])))):.1f} deg")

# Also show what USD viewer typically does: no spawn rot, but may display Y-up
print("\nStage metadata:")
print(f"  upAxis: {UsdGeom.GetStageUpAxis(stage)}")
print(f"  metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)} (sim scales by 0.0254 at spawn)")

# Joint rest info
from pxr import UsdPhysics

for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.RevoluteJoint(prim)
        print("\nRevolute joint at q=0 (authored rest):")
        print(f"  axis: {j.GetAxisAttr().Get()}")
        print(f"  limits: [{j.GetLowerLimitAttr().Get()}, {j.GetUpperLimitAttr().Get()}] deg")
        print(f"  drive targetPosition: {prim.GetAttribute('drive:angular:physics:targetPosition').Get()}")
