"""Sweep spawn rotations and log pegboard/handle orientation."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math

import omni.usd
import torch
from isaaclab.utils.math import quat_from_euler_xyz
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena_environments import lever_scene_builder

ASSET = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
LEVER = "/World/lever_revolute"
BASE = LEVER + "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
HANDLE = (
    BASE
    + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"
)
LAYOUT = LEVER + "/Layout_v9"
pos = lever_scene_builder.LEVER_USD_DEFAULT_POS
scale = lever_scene_builder.LEVER_USD_DEFAULT_SCALE

OUT = open("/workspaces/isaaclab_arena/.lever_tmp/spawn_sweep.log", "w", buffering=1)


def run_case(roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    lever_xf = UsdGeom.Xform.Define(stage, Sdf.Path(LEVER))
    lever_xf.GetPrim().GetReferences().AddReference(ASSET)
    q = quat_from_euler_xyz(
        torch.tensor([math.radians(roll_deg)]),
        torch.tensor([math.radians(pitch_deg)]),
        torch.tensor([math.radians(yaw_deg)]),
    )[0].tolist()
    rot = Gf.Rotation(Gf.Quatd(q[3], Gf.Vec3d(q[0], q[1], q[2])))
    euler = rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    ops = {op.GetOpName(): op for op in lever_xf.GetOrderedXformOps()}
    ops["xformOp:translate"].Set(Gf.Vec3d(*pos))
    ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(euler[0], euler[1], euler[2]))
    ops["xformOp:scale"].Set(Gf.Vec3f(scale, scale, scale))

    from isaacsim.core.api import SimulationContext

    sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0)
    sim.initialize_physics()
    sim.play()
    for _ in range(60):
        sim.step(render=False)

    cache = UsdGeom.XformCache()
    layout_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(LAYOUT))
    handle_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))
    ln = layout_xf.ExtractRotation().TransformDir(Gf.Vec3d(0, 0, 1))
    ln.Normalize()
    bxf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
    off = handle_xf.ExtractTranslation() - bxf.ExtractTranslation()
    off.Normalize()
    OUT.write(
        f"roll={roll_deg:4} pitch={pitch_deg:4} yaw={yaw_deg:4} "
        f"layoutN=({ln[0]:+.2f},{ln[1]:+.2f},{ln[2]:+.2f}) "
        f"handleOff=({off[0]:+.2f},{off[1]:+.2f},{off[2]:+.2f})\n"
    )
    OUT.flush()


for yaw in (0, 90, 180, 270):
    run_case(0, 0, yaw)
for yaw in (0, 90, 180, 270):
    run_case(90, 0, yaw)
for yaw in (0, 90, 180, 270):
    run_case(0, 90, yaw)

OUT.close()
simulation_app.close()
