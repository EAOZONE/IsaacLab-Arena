# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build an articulated USD for the *ws_alex* door and write it under ``assets/doorman/usd``.

The ws_alex door (IHMC ``environmentObjects/door``) ships only as libGDX meshes with no
URDF/USD. This script reproduces it as a single-DOF articulation dimensioned from the
measured ``door_panel.glb`` bounding box (0.9144 m wide x 0.034 m thick x 2.033 m tall,
Z-up, centred on its origin):

* ``frame`` — static door frame (fixed base): two jambs + a lintel.
* ``door``  — the swinging slab plus a lever handle, rigidly attached.
* ``hinge`` — revolute joint, vertical (Z) axis along the hinge-side edge, free-swinging
  (``target_type="none"``) with a little joint friction so it behaves like a real door.

The hinge edge sits at the link-frame origin (x=0); the slab extends to +x and the latch
/ lever live near the +x edge.

Run inside the Arena container::

    /isaac-sim/python.sh isaaclab_arena/scripts/doorman/build_ws_alex_door.py --headless

The output ``ws_alex_door/ws_alex_door.usd`` is consumed by the ``ws_alex_door`` asset
(see ``isaaclab_arena/assets/object_library.py``).
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# --- door geometry, measured from door_panel.glb (metres) ---------------------------------
PANEL_W = 0.9144  # width  (x)
PANEL_T = 0.034  # thickness (y)
PANEL_H = 2.033  # height (z)
HALF_W = PANEL_W / 2.0
HALF_H = PANEL_H / 2.0
FLOOR_GAP = 0.02  # clearance under the slab so it never scrapes the ground (else the hinge jams)
PANEL_CZ = FLOOR_GAP + HALF_H  # slab centre height (bottom sits at FLOOR_GAP)
FRAME_TOP = FLOOR_GAP + PANEL_H + 0.05  # top of the jambs / lintel height
HANDLE_Z = FLOOR_GAP + 1.0  # lever height above floor
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "assets",
    "doorman",
    "usd",
)

ROBOT_NAME = "ws_alex_door"
HINGE_JOINT_NAME = "hinge"


def _box(origin: tuple[float, float, float], size: tuple[float, float, float], material: str) -> str:
    ox, oy, oz = origin
    sx, sy, sz = size
    return (
        f'    <visual><origin xyz="{ox} {oy} {oz}"/><geometry><box size="{sx} {sy} {sz}"/></geometry>'
        f'<material name="{material}"/></visual>\n'
        f'    <collision><origin xyz="{ox} {oy} {oz}"/><geometry><box size="{sx} {sy} {sz}"/></geometry></collision>\n'
    )


def build_urdf() -> str:
    """Return the URDF XML string for the ws_alex door."""
    # frame: hinge-side jamb at x≈0, latch-side jamb past the slab, lintel across the top.
    # Jambs run from the floor (z=0) up past the slab so the static frame sits on the ground.
    jamb = (0.06, 0.10, FRAME_TOP)
    frame_geo = (
        _box((-0.045, 0.0, FRAME_TOP / 2.0), jamb, "frame_mat")
        + _box((PANEL_W + 0.045, 0.0, FRAME_TOP / 2.0), jamb, "frame_mat")
        + _box((HALF_W, 0.0, FRAME_TOP), (PANEL_W + 0.18, 0.10, 0.06), "frame_mat")
    )
    # door: slab centred at +HALF_W from the hinge, raised by FLOOR_GAP so it swings freely.
    door_geo = (
        _box((HALF_W, 0.0, PANEL_CZ), (PANEL_W, PANEL_T, PANEL_H), "door_mat")
        + _box((PANEL_W - 0.08, -0.035, HANDLE_Z), (0.07, 0.05, 0.07), "metal_mat")  # rosette
        + _box((PANEL_W - 0.14, -0.06, HANDLE_Z), (0.14, 0.03, 0.025), "metal_mat")  # lever bar
    )
    return f"""<?xml version="1.0"?>
<robot name="{ROBOT_NAME}">
  <material name="frame_mat"><color rgba="0.40 0.27 0.16 1"/></material>
  <material name="door_mat"><color rgba="0.60 0.42 0.24 1"/></material>
  <material name="metal_mat"><color rgba="0.08 0.08 0.08 1"/></material>

  <link name="frame">
    <inertial>
      <mass value="80.0"/>
      <inertia ixx="2.0" iyy="2.0" izz="2.0" ixy="0" ixz="0" iyz="0"/>
    </inertial>
{frame_geo}  </link>

  <link name="door">
    <inertial>
      <origin xyz="{HALF_W} 0 {PANEL_CZ}"/>
      <mass value="12.0"/>
      <inertia ixx="4.2" iyy="5.0" izz="0.85" ixy="0" ixz="0" iyz="0"/>
    </inertial>
{door_geo}  </link>

  <joint name="{HINGE_JOINT_NAME}" type="revolute">
    <parent link="frame"/>
    <child link="door"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1.92" effort="80" velocity="3"/>
    <dynamics damping="0.2" friction="0.05"/>
  </joint>
</robot>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ws_alex door articulation USD.")
    parser.add_argument("--out_dir", type=str, default=os.path.abspath(DEFAULT_OUT))
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    out_dir = os.path.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    urdf_path = os.path.join(out_dir, f"{ROBOT_NAME}.urdf")
    with open(urdf_path, "w") as f:
        f.write(build_urdf())
    print(f"[doorman] wrote URDF: {urdf_path}")

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=out_dir,
        fix_base=True,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        collision_type="Convex Hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[doorman] generated USD: {converter.usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
