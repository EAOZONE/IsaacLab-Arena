"""Sweep roll/pitch/yaw and report pegboard pitch (angle from vertical)."""

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
LAYOUT = LEVER + "/Layout_v9"
BASE = LEVER + "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
pos = lever_scene_builder.LEVER_USD_DEFAULT_POS
scale = lever_scene_builder.LEVER_USD_DEFAULT_SCALE
yaw = lever_scene_builder.LEVER_USD_DEFAULT_YAW

OUT = open("/workspaces/isaaclab_arena/.lever_tmp/pitch_sweep.log", "w", buffering=1)


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
    for _ in range(30):
        sim.step(render=False)

    cache = UsdGeom.XformCache()
    layout_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(LAYOUT))
    base_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
    ln = layout_xf.ExtractRotation().TransformDir(Gf.Vec3d(0, 0, 1))
    ln.Normalize()
    up = Gf.Vec3d(0, 0, 1)
    # pegboard plane normal tilt from world horizontal (0=vertical wall, 90=flat table)
    tilt_from_vertical = math.degrees(math.acos(min(1.0, max(-1.0, abs(Gf.Dot(ln, up))))))
    base_e = base_xf.ExtractRotation().Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    OUT.write(
        f"roll={roll_deg:5.0f} pitch={pitch_deg:5.0f} yaw={yaw_deg:5.0f} "
        f"layoutN=({ln[0]:+.2f},{ln[1]:+.2f},{ln[2]:+.2f}) "
        f"tiltFromVert={tilt_from_vertical:5.1f} baseEuler={[round(v,1) for v in base_e]}\n"
    )
    OUT.flush()


# current default
run_case(90, 0, yaw)
# nearby pitch fixes
for pitch in (-90, -45, 0, 45, 90):
    for roll in (0, 90):
        run_case(roll, pitch, yaw)
# yaw=0 variants with best-looking rolls
for roll, pitch in [(0, 0), (0, 90), (90, 0), (90, 90), (-90, 0)]:
    run_case(roll, pitch, 0)

OUT.close()
simulation_app.close()
