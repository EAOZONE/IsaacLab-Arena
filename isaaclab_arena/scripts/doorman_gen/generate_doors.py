# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate a curated set of DoorMan articulated door USDs for Arena teleop.

Vendored DoorMan door generator (from GR00T-VisualSim2Real ``origin/doorman``). Each door is an
articulated USD with a ``hinge_joint`` (the openable DOF), a ``handle_joint`` (lever/knob), and an
optional ``latch_joint``; the root carries a ``PhysicsArticulationRootAPI`` so Arena loads it as an
``ARTICULATION`` (consumed by the ``doorman_door`` asset and ``alex_doorman_teleop`` env).

Run once per clone, inside the Arena container (needs Isaac Sim)::

    /isaac-sim/python.sh isaaclab_arena/scripts/doorman_gen/generate_doors.py --num_doors 15

Writes ``door_0000.usd … + metadata.json`` to ``isaaclab_arena/assets/doorman_doors/usd`` (gitignored).
Override the output dir with ``--output_dir`` or ``ARENA_DOORMAN_DOORS_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Vendored modules (door.py / door_builder.py use bare imports) live next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_OUT = os.environ.get(
    "ARENA_DOORMAN_DOORS_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "doorman_doors", "usd")),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num_doors", type=int, default=15, help="Number of door USDs to generate.")
    p.add_argument("--output_dir", type=str, default=_DEFAULT_OUT, help="Output directory for USD files.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    p.add_argument("--build_latch", action=argparse.BooleanOptionalAction, default=True,
                   help="Build a latch that must be released by turning the handle (--no-build_latch to disable).")
    p.add_argument("--add_floors", action="store_true", help="Add a floor plane under each door.")
    p.add_argument("--add_walls", action="store_true", help="Add surrounding walls.")
    p.add_argument("--door_open_lr", nargs="+", default=["right"], choices=["left", "right"],
                   help="Hinge side(s) to sample from.")
    p.add_argument("--door_open_io", nargs="+", default=["out"], choices=["in", "out"],
                   help="Opening direction(s) to sample from.")
    p.add_argument("--door_handle_type", nargs="+", default=["lever"],
                   choices=["knob", "lever", "pushbar", "handle", "flat"], help="Handle type(s) to sample from.")
    p.add_argument("--even_handle_split", action="store_true",
                   help="Assign the given handle types in even contiguous blocks across doors "
                        "(e.g. 14 doors + 'lever pushbar' -> 7 lever then 7 pushbar) instead of "
                        "sampling each door's type randomly.")
    p.add_argument("--door_handle_tblr", nargs=4, type=float, default=[0.95, 0.85, 0.08, 0.15],
                   metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), help="Handle position range (fractions of door).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Launch Isaac Sim headless before any omni/isaaclab/pxr imports.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    import numpy as np
    import omni.usd
    from isaaclab.sim.utils.stage import use_stage
    from pxr import UsdGeom

    from door import DoorSpawnerCfg, _update_joint_transform, build_frame
    from door_builder import _build_door
    from math_utils import set_prim_transform
    from usd_utils import add_collider, add_mass, add_rigid_body, create_plane, create_prim, write_custom_data_to_prim

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = DoorSpawnerCfg(
        build_latch=args.build_latch,
        add_floors=args.add_floors,
        add_walls=args.add_walls,
        door_open_lr=args.door_open_lr,
        door_open_io=args.door_open_io,
        door_handle_type=args.door_handle_type,
        door_handle_tblr=tuple(args.door_handle_tblr),
        randomize_material=False,
    )

    metadata_all: dict = {}
    for i in range(args.num_doors):
        # Even contiguous blocks of handle types (deterministic), e.g. first half lever, second
        # half pushbar. Otherwise _build_door samples each door's type from the full list.
        if args.even_handle_split:
            block = (i * len(args.door_handle_type)) // args.num_doors
            cfg.door_handle_type = [args.door_handle_type[block]]

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # Build at a root-level prim so it can be the stage default prim (USD requires the
        # defaultPrim to be a root prim — Arena's USD spawner references the default prim,
        # mirroring ws_alex_door's ``defaultPrim``).
        prim_path = "/Door"
        # Point isaaclab's stage context at this stage so schemas.modify_* helpers resolve it
        # (get_current_stage() reads isaaclab's own context, not the omni.usd one).
        with use_stage(stage):
            metadata = _build_door(
                stage,
                prim_path,
                cfg,
                material_prim_paths=None,
                _create_prim=create_prim,
                _add_rigid_body=add_rigid_body,
                _add_mass=add_mass,
                _add_collider=add_collider,
                _set_prim_transform=set_prim_transform,
                _create_plane=create_plane,
                _write_custom_data=write_custom_data_to_prim,
                _update_joint_transform=_update_joint_transform,
                _build_frame=build_frame,
            )

        stage.SetDefaultPrim(stage.GetPrimAtPath(prim_path))

        out_path = os.path.abspath(os.path.join(args.output_dir, f"door_{i:04d}.usd"))
        stage.Export(out_path)
        metadata_all[f"door_{i:04d}"] = metadata
        print(f"[viral-door] [{i + 1}/{args.num_doors}] {out_path}")

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_all, f, indent=2, default=str)
    print(f"[viral-door] generated {args.num_doors} doors in {os.path.abspath(args.output_dir)}")

    simulation_app.close()


if __name__ == "__main__":
    main()
