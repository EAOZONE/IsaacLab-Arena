"""Dump LEVER_AGAIN.usd handle geometry: local bbox, joint axis/limits, rest pose."""
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

print("=== All prims ===")
for prim in stage.Traverse():
    print(prim.GetPath(), prim.GetTypeName())

print("\n=== Revolute joints ===")
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsRevoluteJoint":
        j = UsdPhysics.RevoluteJoint(prim)
        print("path:", prim.GetPath())
        print("  axis:", j.GetAxisAttr().Get())
        print("  lowerLimit:", j.GetLowerLimitAttr().Get())
        print("  upperLimit:", j.GetUpperLimitAttr().Get())
        print("  body0:", j.GetBody0Rel().GetTargets())
        print("  body1:", j.GetBody1Rel().GetTargets())
        print("  localPos0:", j.GetLocalPos0Attr().Get())
        print("  localRot0:", j.GetLocalRot0Attr().Get())
        print("  localPos1:", j.GetLocalPos1Attr().Get())
        print("  localRot1:", j.GetLocalRot1Attr().Get())
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            print("  drive target:", drive.GetTargetPositionAttr().Get())
            print("  drive stiffness:", drive.GetStiffnessAttr().Get())
            print("  drive damping:", drive.GetDampingAttr().Get())

print("\n=== Handle_1 rigid body + bbox ===")
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for prim in stage.Traverse():
    if "Handle_1" in str(prim.GetPath()) and prim.GetPath().name == "Handle_1":
        print("Handle_1 prim:", prim.GetPath())
        xform = UsdGeom.Xformable(prim)
        local_to_world = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        print("  local_to_world translation:", local_to_world.ExtractTranslation())
        # local bbox of Handle_1 subtree, in Handle_1's own local frame
        bbox = bbox_cache.ComputeLocalBound(prim)
        rng = bbox.ComputeAlignedRange()
        print("  local bbox min:", rng.GetMin())
        print("  local bbox max:", rng.GetMax())

simulation_app.close()
