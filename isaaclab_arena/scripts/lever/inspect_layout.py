# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Print lever-board prim hierarchy and mesh centers (debug helper)."""

from __future__ import annotations

import argparse
import os

DEFAULT_LAYOUT_USD = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "Lever", "Levers.usd"
)


def _print_subtree(stage, root_path: str, max_depth: int = 8) -> None:
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    assert root.IsValid(), root_path
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    base_depth = len(str(root_path).split("/"))
    for prim in Usd.PrimRange(root):
        depth = len(str(prim.GetPath()).split("/")) - base_depth
        if depth > max_depth:
            continue
        rel = str(prim.GetPath())[len(root_path) :]
        if prim.IsA(UsdGeom.Mesh) or any(
            k in prim.GetName() for k in ("Handle", "Bolt", "Nut", "Stem", "base", "Body")
        ):
            c = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
            print(f"{'  ' * depth}{rel or '/'} {prim.GetTypeName()} center={[round(c[i], 4) for i in range(3)]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_usd", default=os.path.normpath(DEFAULT_LAYOUT_USD))
    parser.add_argument("--assembly", default="Blue_Handled_Valve_v3_1")
    args = parser.parse_args()

    from pxr import Usd

    stage = Usd.Stage.Open(args.layout_usd)
    layout = stage.GetPrimAtPath("/World/Layout_v9")
    print("Top-level assemblies:")
    for child in layout.GetChildren():
        print(" ", child.GetName())
    print(f"\nSubtree {args.assembly}:")
    _print_subtree(stage, f"/World/Layout_v9/{args.assembly}")


if __name__ == "__main__":
    main()
