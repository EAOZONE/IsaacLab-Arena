"""Track Handle_1's simulated orientation over a long physics run, using the real Arena
Object/lever_scene_builder code path, to see whether/when it drifts from its authored pose.
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

from isaaclab_arena_environments import lever_scene_builder

ASSET = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
LEVER = "/World/lever_revolute"
BASE = LEVER + "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
HEX = BASE + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1"
HANDLE = HEX + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"

lever_assets, lever_object = lever_scene_builder.build_lever_scene_assets(
    usd_path=ASSET,
    usd_pos=lever_scene_builder.LEVER_USD_DEFAULT_POS,
    usd_yaw=lever_scene_builder.LEVER_USD_DEFAULT_YAW,
    usd_scale=lever_scene_builder.LEVER_USD_DEFAULT_SCALE,
    lever_dr=False,
    table="none",
)
pose = lever_object.initial_pose
print("pos:", pose.position_xyz, "quat:", pose.rotation_xyzw)
scale = lever_scene_builder.LEVER_USD_DEFAULT_SCALE

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.Scene.Define(stage, "/physicsScene")

light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/_DebugLight"))
light.CreateIntensityAttr(3000.0)
dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/_DebugDome"))
dome.CreateIntensityAttr(1000.0)

lever_xf = UsdGeom.Xform.Define(stage, Sdf.Path(LEVER))
lever_xf.GetPrim().GetReferences().AddReference(ASSET)
q = pose.rotation_xyzw
rot = Gf.Rotation(Gf.Quatd(q[3], Gf.Vec3d(q[0], q[1], q[2])))
euler = rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
ops = {op.GetOpName(): op for op in lever_xf.GetOrderedXformOps()}
ops["xformOp:translate"].Set(Gf.Vec3d(*pose.position_xyz))
ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(euler[0], euler[1], euler[2]))
ops["xformOp:scale"].Set(Gf.Vec3f(scale, scale, scale))

sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0)
sim.initialize_physics()
sim.play()

cache = UsdGeom.XformCache()


def report(label):
    cache.Clear()
    xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))
    rot = xf.ExtractRotation()
    d = rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    pos = xf.ExtractTranslation()
    print(f"{label}: euler=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f}) pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})")


report("t=0 (pre-step)")
checkpoints = [1, 5, 20, 60, 120, 300, 600, 1200]
prev = 0
for cp in checkpoints:
    for _ in range(cp - prev):
        sim.step(render=(cp >= checkpoints[-2]))
    prev = cp
    report(f"t={cp}")

center = np.array(pose.position_xyz) + np.array([0.0, 0.0, 0.1])
set_camera_view(eye=center + np.array([0.9, -0.9, 0.7]), target=center)
for _ in range(60):
    simulation_app.update()

import omni.kit.viewport.utility as vp_util

out = "/workspaces/isaaclab_arena/.lever_tmp/track_settle_final.png"
vp = vp_util.get_active_viewport()
vp_util.capture_viewport_to_file(vp, out)
for _ in range(60):
    simulation_app.update()
print("saved", out)

simulation_app.close()
