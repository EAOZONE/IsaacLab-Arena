"""Dump prim hierarchy + physics info from a USD file."""
import sys

from pxr import Usd, UsdGeom, UsdPhysics

path = sys.argv[1]
stage = Usd.Stage.Open(path)
print("upAxis:", UsdGeom.GetStageUpAxis(stage))
print("metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))
print("defaultPrim:", stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None)
print()

for prim in stage.Traverse():
    depth = str(prim.GetPath()).count("/") - 1
    apis = prim.GetAppliedSchemas()
    api_str = (" APIs=" + ",".join(apis)) if apis else ""
    inst = " INSTANCE" if prim.IsInstance() else ""
    print("  " * depth + f"{prim.GetName()} [{prim.GetTypeName()}]{api_str}{inst}")
    # references / payloads
    md = prim.GetMetadata("references")
    if md:
        for it in md.ApplyOperations([]):
            print("  " * depth + f"   -> ref: {it.assetPath} {it.primPath}")
    pl = prim.GetMetadata("payload")
    if pl:
        for it in pl.ApplyOperations([]):
            print("  " * depth + f"   -> payload: {it.assetPath} {it.primPath}")
    # joints
    if prim.IsA(UsdPhysics.Joint):
        j = UsdPhysics.Joint(prim)
        print("  " * depth + f"   body0={j.GetBody0Rel().GetTargets()} body1={j.GetBody1Rel().GetTargets()}")
        for attr in prim.GetAttributes():
            if attr.HasAuthoredValue():
                print("  " * depth + f"   {attr.GetName()} = {attr.Get()}")
