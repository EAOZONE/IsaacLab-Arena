"""Spawn the lever at the scene-builder pose, play physics, report handle drift from editor pose."""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import math
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from isaaclab_arena_environments import lever_scene_builder

ASSET = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
LEVER = "/World/lever_revolute"
BASE = LEVER + "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
HANDLE = BASE + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"

pos = lever_scene_builder.LEVER_USD_DEFAULT_POS
yaw = lever_scene_builder.LEVER_USD_DEFAULT_YAW
scale = lever_scene_builder.LEVER_USD_DEFAULT_SCALE

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.Scene.Define(stage, "/physicsScene")
lever_xf = UsdGeom.Xform.Define(stage, Sdf.Path(LEVER))
lever_xf.GetPrim().GetReferences().AddReference(ASSET)
ops = {op.GetOpName(): op for op in lever_xf.GetOrderedXformOps()}
ops["xformOp:translate"].Set(Gf.Vec3d(*pos))
ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, yaw))
ops["xformOp:scale"].Set(Gf.Vec3f(scale, scale, scale))

def handle_rot():
    cache = UsdGeom.XformCache()
    return cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE)).ExtractRotation()

def handle_dir():
    cache = UsdGeom.XformCache()
    m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))
    hexm = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
    return m

rot_before = handle_rot()

from isaacsim.core.api import SimulationContext
sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=1.0 / 200.0, rendering_dt=1.0 / 200.0)
sim.initialize_physics()
sim.play()
import time
checkpoints = {10, 60, 120, 240}
for i in range(1, 241):
    sim.step(render=False)
    if i in checkpoints:
        delta = (handle_rot() * rot_before.GetInverse())
        print(f"step {i:4d}: handle drift from editor pose = {delta.GetAngle():7.3f} deg (axis {delta.GetAxis()})")
simulation_app.close()
