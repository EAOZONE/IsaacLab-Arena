"""Inspect Lever_revolute.usd joint and handle orientation."""

from pxr import Gf, Usd, UsdGeom, UsdPhysics

path = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
stage = Usd.Stage.Open(path)
print("defaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print("upAxis:", UsdGeom.GetStageUpAxis(stage))
print("metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))

world = stage.GetPrimAtPath("/World")
if world:
    xfc = UsdGeom.Xformable(world)
    for op in xfc.GetOrderedXformOps():
        print(f"/World {op.GetOpName()}: {op.Get()}")

for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        print("\nRevoluteJoint:", prim.GetPath())
        joint = UsdPhysics.RevoluteJoint(prim)
        for attr in prim.GetAttributes():
            name = attr.GetName()
            val = attr.Get()
            if val is not None and not name.startswith("xformOp"):
                print(f"  {name}: {val}")

BASE = (
    "/World/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
)
HANDLE = (
    BASE
    + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"
)

cache = UsdGeom.XformCache()
for label, hp in [("base_link", BASE), ("Handle_1", HANDLE)]:
    p = stage.GetPrimAtPath(hp)
    if not p:
        print(f"MISSING {hp}")
        continue
    xf = cache.GetLocalToWorldTransform(p)
    rot = xf.ExtractRotation()
    decomp = rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    print(f"\n{label} world pos: {xf.ExtractTranslation()}")
    print(f"  rot XYZ (deg): ({decomp[0]:.1f}, {decomp[1]:.1f}, {decomp[2]:.1f})")

# Handle relative to base
base_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
handle_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))
rel = handle_xf * base_xf.GetInverse()
rel_rot = rel.ExtractRotation().Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
print(f"\nHandle relative to base rot XYZ (deg): ({rel_rot[0]:.1f}, {rel_rot[1]:.1f}, {rel_rot[2]:.1f})")

# Simulate spawn rotation from lever_scene_builder (roll=pi/2, yaw=180 deg)
import math

import torch
from isaaclab.utils.math import quat_from_euler_xyz

lever_yaw_rad = math.radians(90.0 + 90.0)  # usd_yaw default 90
spawn_quat = quat_from_euler_xyz(
    torch.tensor([math.pi / 2.0]),
    torch.tensor([0.0]),
    torch.tensor([lever_yaw_rad]),
)[0].tolist()
print(f"\nSim spawn quat (xyzw): {spawn_quat}")

# Apply spawn rotation to handle world transform
spawn_rot = Gf.Rotation(Gf.Quatd(spawn_quat[3], Gf.Vec3d(spawn_quat[0], spawn_quat[1], spawn_quat[2])))
spawned_handle_rot = spawn_rot * handle_xf.ExtractRotation()
decomp2 = spawned_handle_rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
print(f"Handle after sim spawn rot XYZ (deg): ({decomp2[0]:.1f}, {decomp2[1]:.1f}, {decomp2[2]:.1f})")
