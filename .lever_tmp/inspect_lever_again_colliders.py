"""Which meshes under Handle_1 actually have collision, and where do they live in world space
once spawned live in the alex_empty scene (accounting for the 0.0254 spawn scale)?
"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--viz", "none"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

usd_path = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd"
ctx = omni.usd.get_context()
ctx.open_stage(usd_path)
stage = ctx.get_stage()

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for prim in stage.Traverse():
    if "Handle_1" not in str(prim.GetPath()):
        continue
    has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    if prim.GetTypeName() == "Mesh" or has_collision or has_rigid:
        xform = UsdGeom.Xformable(prim)
        world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_tf.ExtractTranslation()
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        print(
            f"{prim.GetPath()} type={prim.GetTypeName()} collision={has_collision} rigid={has_rigid}\n"
            f"    world_translation={tuple(translation)}\n"
            f"    world_bbox_min={tuple(rng.GetMin())} world_bbox_max={tuple(rng.GetMax())}"
        )

simulation_app.close()
