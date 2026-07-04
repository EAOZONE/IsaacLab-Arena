"""Dump local + world transforms for prims under a subtree, and all reference asset paths."""
import sys

from pxr import Usd, UsdGeom

path = sys.argv[1]
stage = Usd.Stage.Open(path)
cache = UsdGeom.XformCache()

roots = [
    "/World/Layout_v9/Blue_Handled_Valve_v3_1",
    "/World/Layout_v9/Red_Handled_Valve_v4_1",
    "/World/Layout_v9/Radiator_Cap_v5_1",
    "/World/Layout_v9/Dipstick_Interface_v1_1",
]
for root in roots:
    rp = stage.GetPrimAtPath(root)
    if not rp:
        continue
    print("=" * 80)
    for prim in Usd.PrimRange(rp):
        if prim.GetTypeName() not in ("Xform", "Mesh"):
            continue
        x = UsdGeom.Xformable(prim)
        ops = [(op.GetOpName(), op.Get()) for op in x.GetOrderedXformOps()]
        wt = cache.GetLocalToWorldTransform(prim)
        t = wt.ExtractTranslation()
        print(prim.GetPath())
        for name, val in ops:
            print(f"    {name} = {val}")
        print(f"    world_pos = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
print("=" * 80)
print("all reference asset paths:")
seen = set()
for prim in stage.TraverseAll():
    md = prim.GetMetadata("references")
    if md:
        for it in md.ApplyOperations([]):
            if it.assetPath not in seen:
                seen.add(it.assetPath)
                print(" ", prim.GetPath(), "->", it.assetPath)
