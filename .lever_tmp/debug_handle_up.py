"""Debug handle orientation at spawn with current lever_scene_builder defaults."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import math
import traceback

import omni.usd
import torch
from isaaclab.utils.math import quat_from_euler_xyz
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena_environments import lever_scene_builder

ASSET = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
LEVER = "/World/lever_revolute"
BASE = LEVER + "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
HANDLE = BASE + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"

pos = lever_scene_builder.LEVER_USD_DEFAULT_POS
yaw = lever_scene_builder.LEVER_USD_DEFAULT_YAW
scale = lever_scene_builder.LEVER_USD_DEFAULT_SCALE

OUT = open("/workspaces/isaaclab_arena/.lever_tmp/spawn_debug.log", "w", buffering=1)


def log(msg: str) -> None:
    OUT.write(msg + "\n")
    OUT.flush()


def handle_dirs(stage, label: str) -> None:
    cache = UsdGeom.XformCache()
    bxf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
    hxf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))
    rot = hxf.ExtractRotation()
    off = hxf.ExtractTranslation() - bxf.ExtractTranslation()
    off.Normalize()
    for axis_name, local in [("+X", Gf.Vec3d(1, 0, 0)), ("+Y", Gf.Vec3d(0, 1, 0)), ("+Z", Gf.Vec3d(0, 0, 1))]:
        d = rot.TransformDir(local)
        d.Normalize()
        log(f"{label} handle local {axis_name} -> world ({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})")
    log(f"{label} base->handle offset ({off[0]:+.2f},{off[1]:+.2f},{off[2]:+.2f})")


try:
    quat = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([math.radians(yaw)]))[0].tolist()
    log(f"defaults pos={pos} yaw={yaw} scale={scale}")

    for target in (65.0, 0.0, -65.0):
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        lever_xf = UsdGeom.Xform.Define(stage, Sdf.Path(LEVER))
        lever_xf.GetPrim().GetReferences().AddReference(ASSET)
        rot = Gf.Rotation(Gf.Quatd(quat[3], Gf.Vec3d(quat[0], quat[1], quat[2])))
        euler = rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
        ops = {op.GetOpName(): op for op in lever_xf.GetOrderedXformOps()}
        ops["xformOp:translate"].Set(Gf.Vec3d(*pos))
        ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(euler[0], euler[1], euler[2]))
        ops["xformOp:scale"].Set(Gf.Vec3f(scale, scale, scale))
        joint = stage.GetPrimAtPath(BASE + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/RevoluteJoint")
        joint.GetAttribute("drive:angular:physics:targetPosition").Set(target)

        from isaacsim.core.api import SimulationContext

        sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0)
        sim.initialize_physics()
        sim.play()
        for _ in range(120):
            sim.step(render=False)
        log(f"\n=== targetPosition={target} ===")
        handle_dirs(stage, "sim")

    # native USD at q=0 reference (no sim, Y-up file)
    stage = Usd.Stage.Open(ASSET)
    cache = UsdGeom.XformCache()
    bxf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE.replace(LEVER + "/", "/World/")))
    hxf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE.replace(LEVER + "/", "/World/")))
    off = hxf.ExtractTranslation() - bxf.ExtractTranslation()
    off.Normalize()
    log("\n=== native USD q=0 (Y-up file) ===")
    log(f"base->handle offset ({off[0]:+.2f},{off[1]:+.2f},{off[2]:+.2f})")

except Exception:
    log(traceback.format_exc())
finally:
    OUT.close()
    simulation_app.close()
