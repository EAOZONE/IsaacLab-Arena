# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Alex + lever demonstrations from a scripted Cartesian path, not teleop.

The lever's live world pose is read straight out of the scene (no human
operator needed). The scripted motion closes the hand into a thumbs-up fist
(fingers curled, thumb left extended and clear of the contact surface), moves
above the lever, then pushes straight down through the lever's range --
rather than grasping the handle and rotating the wrist to match it -- since a
fixed-fist push is an easier target for a policy to learn than a precision
grasp. The
resulting per-step actions are recorded through the same hdf5 pipeline
``record_demos.py`` uses -- so the output dataset is a drop-in for existing
GR00T / CCIL training.

Only ``alex_v2_ability_hands`` (Pink IK, world-frame wrist targets + raw hand
joints) is supported: the path is expressed as absolute end-effector poses,
which that action term consumes directly. Only single-env (``--num_envs 1``)
runs are supported -- the arm-path math below is unbatched.

The push geometry (where above the lever to press, how far to stand off before
descending) is **not** derived from the mesh -- it is a CLI-tunable guess, in
the same spirit as the board placement in ``alex_empty_environment.py``. The
wrist keeps its starting orientation throughout (no rotation needed to press
down) and "up"/"down" are world Z, not the lever's own (possibly tilted/yawed)
frame. Run once with ``--enable_cameras --viz kit`` to watch the motion in the
GUI and adjust ``--push_local_offset`` / ``--approach_height`` /
``--push_wrist_rot_offset`` before recording for real.

Alongside the standard Arena hdf5, this also writes a companion hdf5
(``--lever_eef_dataset_file``) with ``observation.state``/``action`` arrays
packed into the same layout as the real-hardware
`H2Ozone/test_obs_new <https://huggingface.co/datasets/H2Ozone/test_obs_new>`_
dataset (see its ``meta/info.json`` and
``isaaclab_arena_gr00t/embodiments/alex/alex_test_obs_new_modality.json``):
48-dim state ``[left_wrist_pose(7), right_wrist_pose(7), left_forearm_quat(4),
right_forearm_quat(4), head_quat(4), left_hand(10), right_hand(10),
spine(2)]`` and 46-dim action with the same fields minus spine. Wrist poses are
expressed in the same env/world frame as the referenced dataset. The script does
not command forearm/head/spine directly, so those columns use the robot's actual
live orientation/joint readings while the wrist and hand action columns use the
scripted targets.
Turning this into a full LeRobot v3 dataset (parquet + mp4 + meta) still needs
video export -- that part is unrelated to the schema and isn't done here.

Run inside the container::

    /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/record_scripted_lever_demos.py \\
        --viz kit --dataset_file /datasets/lever_scripted.hdf5 --num_demos 5 \\
        alex_empty --embodiment alex_v2_ability_hands \\
        --usd isaaclab_arena/assets/lever_sim/Lever_revolute.usd
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--dataset_file",
    type=str,
    required=True,
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--step_hz", type=int, default=30, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record."
)
parser.add_argument(
    "--object_name",
    type=str,
    default="lever_revolute",
    help="Scene key of the lever rigid object to reach for (default matches alex_empty's"
    " Lever_revolute.usd asset name).",
)
parser.add_argument(
    "--arm",
    type=str,
    choices=["left", "right"],
    default="right",
    help="Which arm pushes the lever.",
)
parser.add_argument(
    "--push_local_offset",
    type=lambda arg: [float(v) for v in arg.split(",")],
    default=[-0.055, 0.0, 0.0],
    help="Press point x,y,z [m] in the lever handle's own local (rest-pose) frame -- only the"
    " horizontal placement of the contact point over the lever (untuned guess -- verify visually).",
)
parser.add_argument(
    "--push_wrist_rot_offset",
    type=lambda arg: [float(v) for v in arg.split(",")],
    default=[0.0, 0.0, 0.0, 1.0],
    help="Extra hand orientation x,y,z,w applied on top of the arm's own starting (home) orientation --"
    " held constant through the whole motion. Identity (default) keeps the wrist exactly as it starts;"
    " this is NOT relative to the lever's orientation, since a press doesn't need the wrist to track it.",
)
parser.add_argument(
    "--approach_height",
    type=float,
    default=0.08,
    help="Standoff height [m] straight up in world Z above the press point (not the lever's local frame,"
    " which may be tilted/yawed -- e.g. with --lever_dr).",
)
parser.add_argument(
    "--push_target_deg",
    type=float,
    default=70.0,
    help="How far to move the lever through its range (stays clear of the hard limit). Used only"
    " to size the push segment's straight-line displacement via the hinge geometry -- the hand"
    " still moves in a straight line from the contact point (see --min_push_depth), not along the"
    " lever's own arc. Direction follows the joint's actual hinge axis, read from the USD (straight"
    " down for a horizontal axis, a horizontal swipe for a vertical one).",
)
parser.add_argument(
    "--min_push_depth",
    type=float,
    default=0.03,
    help="Floor [m] on the push segment's displacement magnitude computed from --push_target_deg,"
    " in case the hinge-geometry estimate comes out too small to actually move the lever.",
)
parser.add_argument(
    "--close_fraction",
    type=float,
    default=1.0,
    help="How far to close the hand into a fist.",
)
parser.add_argument(
    "--hold_steps",
    type=int,
    default=15,
    help="Steps to settle at the arm's starting pose.",
)
parser.add_argument(
    "--close_steps",
    type=int,
    default=20,
    help="Steps to close the hand into a fist at the home pose.",
)
parser.add_argument(
    "--approach_steps",
    type=int,
    default=45,
    help="Steps from home to the standoff above the lever.",
)
parser.add_argument(
    "--descend_steps",
    type=int,
    default=30,
    help="Steps from the standoff straight down onto the contact point, before pushing/swiping.",
)
parser.add_argument(
    "--push_steps",
    type=int,
    default=90,
    help="Steps to move through --push_target_deg once at the contact point -- direction follows"
    " the joint's own hinge axis (straight down for a horizontal axis, a horizontal swipe for a"
    " vertical one; see --min_push_depth for the general 'push distance' floor).",
)
parser.add_argument(
    "--dwell_steps", type=int, default=15, help="Steps to hold the pressed-down pose."
)
parser.add_argument(
    "--retreat_steps",
    type=int,
    default=30,
    help="Steps back up to the standoff above the lever.",
)
parser.add_argument(
    "--release_steps",
    type=int,
    default=20,
    help="Steps to open the hand at the standoff.",
)
parser.add_argument(
    "--return_steps",
    type=int,
    default=30,
    help="Steps back to the arm's starting pose.",
)
parser.add_argument(
    "--lever_eef_dataset_file",
    type=str,
    default=None,
    help="Also write observation.state/action arrays matching the H2Ozone/test_obs_new 48/46-dim schema"
    " to this hdf5. Defaults to --dataset_file with a '_lever_eef' suffix; pass 'none' to skip.",
)
# NOTE(alexmillane, 2025.09.04): This has to be added last, because
# of the app specific flags being parsed after the global flags.
add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import h5py
import os
import torch

import warp as wp
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

from isaaclab_arena.embodiments.alex.alex import (
    ABILITY_HAND_TELEOP_JOINT_ORDER,
    ALEX_ABILITY_HAND_WRIST_ACTION_DIM,
    build_ability_hand_joint_action,
    build_ability_hand_thumbs_up_action,
)
from isaaclab_arena.utils.cartesian_waypoints import LinearSegment, play_segments
from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg

# H2Ozone/test_obs_new's hand-joint layout groups per finger (q1, q2) and per side, unlike the
# Pink IK action term's interleaved ABILITY_HAND_TELEOP_JOINT_ORDER. See that dataset's
# meta/info.json and isaaclab_arena_gr00t/embodiments/alex/alex_test_obs_new_modality.json.
_LEVER_EEF_HAND_JOINT_ORDER = [
    f"{side}_ability_hand_{suffix}"
    for side in ("left", "right")
    for suffix in (
        "index_q1",
        "index_q2",
        "middle_q1",
        "middle_q2",
        "ring_q1",
        "ring_q2",
        "pinky_q1",
        "pinky_q2",
        "thumb_q1",
        "thumb_q2",
    )
]
_PINK_IK_TO_LEVER_EEF_HAND_PERM = [
    ABILITY_HAND_TELEOP_JOINT_ORDER.index(name) for name in _LEVER_EEF_HAND_JOINT_ORDER
]
_LEVER_EEF_SPINE_JOINT_NAMES = ["SPINE_Z", "SPINE_Y"]
_LEVER_EEF_LEFT_FOREARM_LINK = "LEFT_WRIST_Z_LINK"
_LEVER_EEF_RIGHT_FOREARM_LINK = "RIGHT_WRIST_Z_LINK"
_LEVER_EEF_HEAD_LINK = "HEAD_LINK"
_DEFAULT_PUSH_LOCAL_OFFSET = [-0.055, 0.0, 0.0]
# push_local_offset places the *wrist*, not the fist -- but the closed ability-hand fingers sit
# ~10-12cm forward of RIGHT_GRIPPER_Z_LINK (measured live: right_ability_hand_*_L1/L2 body
# positions), so naively aiming the wrist at the handle overshoots the fingers well past it. This
# is the wrist target that lands the fist's fingertip centroid on Handle_1's collision-mesh bbox
# center (Body1 + Body2, measured in Handle_1's own local frame -- static, safe to compute from
# the raw asset file -- then composed with the *live* RigidPrim-read handle pose, after
# _lever_handle_prim_pose's stale-USD-read fix; see .lever_tmp/ for the derivation scripts).
# Expressed in the handle's local (rest-pose) frame.
_LEVER_AGAIN_PUSH_LOCAL_OFFSET = [0.03603, 0.03933, 0.10620]


def _create_environment():
    """Build the (unwrapped) env and wire up the hdf5 recorder, matching record_demos.py."""
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
        env_cfg.num_rerenders_on_reset = 3
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir or "."
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    import gymnasium as gym

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    reapply_viewer_cfg(env)
    return env


def _body_pose(env, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live (pos, quat_xyzw) of ``link_name`` on the robot, relative to its env origin."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies([link_name])
    idx = int(body_ids[0])
    pos = wp.to_torch(robot.data.body_pos_w)[0, idx] - env.scene.env_origins[0]
    quat = wp.to_torch(robot.data.body_quat_w)[0, idx]
    return pos.clone(), quat.clone()


def _wrist_pose(env, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live (pos, quat_xyzw) of a wrist/gripper link, relative to its env origin."""
    return _body_pose(env, link_name)


def _pose_vector(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Pack an env/world-frame pose as ``[xyz, quat_xyzw]``."""
    return torch.cat([pos, quat])


def _body_quat(env, link_name: str) -> torch.Tensor:
    """Live body orientation quaternion in the env/world frame."""
    _, quat = _body_pose(env, link_name)
    return quat.clone()


def _lever_handle_prim_pose(env, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live Handle_1 prim pose for lever USDs spawned as static/base assets.

    Uses ``isaacsim.core.prims.RigidPrim`` (Fabric-synced), not raw
    ``UsdGeom.Xformable.ComputeLocalToWorldTransform`` -- the latter silently returns the
    USD-authored (design-time) transform and never reflects PhysX simulation results at all
    (confirmed: 300 steps under active gravity produced a bit-identical pose via the Xformable
    read, while RigidPrim showed continuous drift over the same run). This was the actual reason
    every scripted push registered exactly 0.00 deg of handle motion regardless of contact
    accuracy -- the "live" pose being compared against was frozen at spawn.
    """
    from isaacsim.core.prims import RigidPrim

    from isaaclab_arena_environments.lever_scene_builder import (
        LEVER_HANDLE_RIGID_BODY_SUFFIX,
    )

    prim_path = f"/World/envs/env_0/{object_name}{LEVER_HANDLE_RIGID_BODY_SUFFIX}"
    pos_w, quat_wxyz = RigidPrim(prim_path).get_world_poses()
    pos = pos_w[0].to(device=env.device, dtype=env.scene.env_origins.dtype)
    # RigidPrim is scalar-first (w, x, y, z); the rest of this codebase uses xyzw.
    quat = quat_wxyz[0][[1, 2, 3, 0]].to(device=env.device, dtype=pos.dtype)
    return pos - env.scene.env_origins[0], quat


def _object_pose(env, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live (pos, quat_xyzw) of the lever handle, relative to its env origin."""
    assert object_name in env.scene.keys(), (
        f"'{object_name}' not found in the scene (available: {list(env.scene.keys())}). Pass --object_name to match"
        " the --usd asset's prim name."
    )
    data = getattr(env.scene[object_name], "data", None)
    if data is None or not hasattr(data, "root_pos_w"):
        return _lever_handle_prim_pose(env, object_name)

    pos = wp.to_torch(data.root_pos_w)[0] - env.scene.env_origins[0]
    quat = wp.to_torch(data.root_quat_w)[0]
    return pos.clone(), quat.clone()


def _build_lever_path(env, device: torch.device):
    """Close hand -> above the lever -> descend onto the contact point -> push/swipe -> dwell ->
    retreat -> release -> home.

    Wrist orientation is decoupled from the lever's own orientation the whole time: the fist keeps
    its starting (home) orientation -- no wrist rotation is scripted -- and the approach/descend
    phases move in literal world Z, not the handle's local frame (which is tilted/yawed,
    especially with ``--lever_dr``; rotating those offsets through it was sending the hand
    sideways instead of down). The push/swipe phase's *direction*, unlike approach/descend, is not
    fixed to world Z -- it follows the joint's actual hinge axis (read from the USD via
    ``_lever_joint_axis_local``), since a straight-down push only produces torque about a
    horizontal axis; a vertical-axis lever (e.g. LEVER_AGAIN.usd) needs a horizontal swipe
    instead. Only the horizontal (x, y) placement of the contact point tracks the lever's live
    pose, since that's what determines *where* on the lever to press.
    """
    import math

    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    gripper_link = f"{args_cli.arm.upper()}_GRIPPER_Z_LINK"
    home_pos, home_quat = _wrist_pose(env, gripper_link)
    handle_pos, handle_quat = _object_pose(env, args_cli.object_name)

    push_local_offset_values = args_cli.push_local_offset
    if (
        args_cli.object_name == "lever_again"
        and push_local_offset_values == _DEFAULT_PUSH_LOCAL_OFFSET
    ):
        push_local_offset_values = _LEVER_AGAIN_PUSH_LOCAL_OFFSET
    push_local_offset = torch.tensor(
        push_local_offset_values, device=device, dtype=home_pos.dtype
    )
    push_wrist_rot_offset = torch.tensor(
        args_cli.push_wrist_rot_offset, device=device, dtype=home_quat.dtype
    )
    push_quat = quat_mul(
        home_quat.unsqueeze(0), push_wrist_rot_offset.unsqueeze(0)
    ).squeeze(0)

    # push_local_offset is the wrist's own target position, in the handle's local (rest-pose)
    # frame -- found interactively with tune_lever_push_pose.py, which jogs the wrist itself, so
    # no separate fist-knuckle correction is needed here.
    wrist_contact_pos = handle_pos + quat_apply(
        handle_quat.unsqueeze(0), push_local_offset.unsqueeze(0)
    ).squeeze(0)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=wrist_contact_pos.dtype)
    above_pos = wrist_contact_pos + args_cli.approach_height * world_up

    # Push-segment displacement: how far (and which straight-line direction) the contact point
    # would move if the lever rotated through --push_target_deg about its *actual* hinge axis
    # (read from the USD, not assumed) -- the hand still travels in a straight line from the
    # contact point, not along this arc, but the direction now generalizes to whatever the hinge
    # axis actually is: straight down for a horizontal axis (the old Lever_revolute.usd design),
    # a horizontal swipe for a vertical one (e.g. LEVER_AGAIN.usd) -- a purely vertical push force
    # produces zero torque about a vertical axis, so "down" only works when the axis is horizontal.
    joint_axis_local = torch.tensor(
        [_lever_joint_axis_local(args_cli.object_name)], device=device, dtype=handle_quat.dtype
    )
    axis_world = quat_apply(handle_quat.unsqueeze(0), joint_axis_local).squeeze(0)
    push_angle = torch.tensor(
        [math.radians(args_cli.push_target_deg)], device=device, dtype=axis_world.dtype
    )
    push_rot = quat_from_angle_axis(push_angle, axis_world.unsqueeze(0)).squeeze(0)
    rotated_contact_pos = handle_pos + quat_apply(
        push_rot.unsqueeze(0), (wrist_contact_pos - handle_pos).unsqueeze(0)
    ).squeeze(0)
    push_delta = rotated_contact_pos - wrist_contact_pos
    push_delta_norm = float(torch.linalg.norm(push_delta))
    if push_delta_norm < 1e-6:
        # Degenerate (near-zero) estimate -- fall back to the old straight-down convention rather
        # than dividing by ~0.
        push_delta = -args_cli.min_push_depth * world_up
    elif push_delta_norm < args_cli.min_push_depth:
        push_delta = push_delta * (args_cli.min_push_depth / push_delta_norm)
    pushed_pos = wrist_contact_pos + push_delta

    left_close = args_cli.close_fraction if args_cli.arm == "left" else 0.0
    right_close = args_cli.close_fraction if args_cli.arm == "right" else 0.0
    open_hand = build_ability_hand_joint_action(0.0, 0.0, device=device)
    closed_hand = build_ability_hand_thumbs_up_action(
        left_close, right_close, device=device
    )

    return [
        LinearSegment(
            home_pos,
            home_quat,
            open_hand,
            home_pos,
            home_quat,
            open_hand,
            args_cli.hold_steps,
        ),
        LinearSegment(
            home_pos,
            home_quat,
            open_hand,
            home_pos,
            home_quat,
            closed_hand,
            args_cli.close_steps,
        ),
        LinearSegment(
            home_pos,
            home_quat,
            closed_hand,
            above_pos,
            push_quat,
            closed_hand,
            args_cli.approach_steps,
        ),
        LinearSegment(
            above_pos,
            push_quat,
            closed_hand,
            wrist_contact_pos,
            push_quat,
            closed_hand,
            args_cli.descend_steps,
        ),
        LinearSegment(
            wrist_contact_pos,
            push_quat,
            closed_hand,
            pushed_pos,
            push_quat,
            closed_hand,
            args_cli.push_steps,
        ),
        LinearSegment(
            pushed_pos,
            push_quat,
            closed_hand,
            pushed_pos,
            push_quat,
            closed_hand,
            args_cli.dwell_steps,
        ),
        LinearSegment(
            pushed_pos,
            push_quat,
            closed_hand,
            above_pos,
            push_quat,
            closed_hand,
            args_cli.retreat_steps,
        ),
        LinearSegment(
            above_pos,
            push_quat,
            closed_hand,
            above_pos,
            push_quat,
            open_hand,
            args_cli.release_steps,
        ),
        LinearSegment(
            above_pos,
            push_quat,
            open_hand,
            home_pos,
            home_quat,
            open_hand,
            args_cli.return_steps,
        ),
    ]


def _bimanual_wrist_targets(
    env, active_arm: str, pos: torch.Tensor, quat: torch.Tensor
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Wrist targets for both arms: ``active_arm`` gets (pos, quat); the other holds its current pose."""
    idle_arm = "left" if active_arm == "right" else "right"
    idle_pos, idle_quat = _wrist_pose(env, f"{idle_arm.upper()}_GRIPPER_Z_LINK")
    return {active_arm: (pos, quat), idle_arm: (idle_pos, idle_quat)}


def _pink_ik_action(
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]], hand: torch.Tensor
) -> torch.Tensor:
    """Assemble the 34-D ability-hand Pink IK action from bimanual wrist targets + hand joints."""
    left_pos, left_quat = targets["left"]
    right_pos, right_quat = targets["right"]
    return torch.cat([left_pos, left_quat, right_pos, right_quat, hand])


def _hand_joint_pos(env) -> torch.Tensor:
    """Live ability-hand joint positions, in ``ABILITY_HAND_TELEOP_JOINT_ORDER``."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(
        ABILITY_HAND_TELEOP_JOINT_ORDER, preserve_order=True
    )
    return wp.to_torch(robot.data.joint_pos)[0, joint_ids].clone()


def _joint_pos_or_zero(env, joint_names: list[str]) -> torch.Tensor:
    """Live joint positions in order, using zero for joints absent from this embodiment."""
    robot = env.scene["robot"]
    joint_name_to_id = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    joint_pos = wp.to_torch(robot.data.joint_pos)[0]
    values = [
        (
            joint_pos[joint_name_to_id[name]]
            if name in joint_name_to_id
            else joint_pos.new_zeros(())
        )
        for name in joint_names
    ]
    return torch.stack(values).clone()


def _grouped_hand_joints(hand_pink_ik_order: torch.Tensor) -> torch.Tensor:
    """Convert Pink IK's interleaved hand slots to the test_obs_new per-finger grouping."""
    return hand_pink_ik_order[_PINK_IK_TO_LEVER_EEF_HAND_PERM]


def _lever_eef_state_vector(env) -> torch.Tensor:
    """Pack live robot state into the H2Ozone/test_obs_new 48-dim layout."""
    left_pos, left_quat = _wrist_pose(env, "LEFT_GRIPPER_Z_LINK")
    right_pos, right_quat = _wrist_pose(env, "RIGHT_GRIPPER_Z_LINK")
    return torch.cat(
        [
            _pose_vector(left_pos, left_quat),
            _pose_vector(right_pos, right_quat),
            _body_quat(env, _LEVER_EEF_LEFT_FOREARM_LINK),
            _body_quat(env, _LEVER_EEF_RIGHT_FOREARM_LINK),
            _body_quat(env, _LEVER_EEF_HEAD_LINK),
            _grouped_hand_joints(_hand_joint_pos(env)),
            _joint_pos_or_zero(env, _LEVER_EEF_SPINE_JOINT_NAMES),
        ]
    )


def _lever_eef_action_vector(
    env, targets: dict[str, tuple[torch.Tensor, torch.Tensor]], hand: torch.Tensor
) -> torch.Tensor:
    """Pack scripted targets into the H2Ozone/test_obs_new 46-dim action layout."""
    return torch.cat(
        [
            _pose_vector(*targets["left"]),
            _pose_vector(*targets["right"]),
            _body_quat(env, _LEVER_EEF_LEFT_FOREARM_LINK),
            _body_quat(env, _LEVER_EEF_RIGHT_FOREARM_LINK),
            _body_quat(env, _LEVER_EEF_HEAD_LINK),
            _grouped_hand_joints(hand),
        ]
    )


def export_episode_as_success(env) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes(
        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
    )
    env.recorder_manager.export_episodes([0])


def _find_lever_revolute_joint_prim(object_name: str):
    """Find the (single) PhysicsRevoluteJoint prim under a lever scene object."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    candidates = [
        prim
        for prim in stage.Traverse()
        if prim.GetTypeName() == "PhysicsRevoluteJoint"
        and f"/{object_name}/" in str(prim.GetPath())
    ]
    assert len(candidates) == 1, (
        f"Expected one revolute joint under scene object {object_name!r}, found "
        f"{[str(prim.GetPath()) for prim in candidates]}"
    )
    return candidates[0]


_JOINT_AXIS_TOKEN_TO_LOCAL_VEC = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}


def _lever_joint_axis_local(object_name: str) -> list[float]:
    """Return the revolute joint's own hinge axis as a local unit vector, read from the USD.

    The push-depth estimate needs the real hinge axis to rotate the contact point about -- this
    varies per lever asset (LEVER_AGAIN.usd's joint uses Z; Lever_revolute.usd's uses Y), so it
    must be read from ``physics:axis`` rather than assumed.
    """
    joint = _find_lever_revolute_joint_prim(object_name)
    axis_token = str(joint.GetAttribute("physics:axis").Get())
    assert axis_token in _JOINT_AXIS_TOKEN_TO_LOCAL_VEC, (
        f"Unexpected physics:axis {axis_token!r} on {joint.GetPath()}"
    )
    return _JOINT_AXIS_TOKEN_TO_LOCAL_VEC[axis_token]


def _lever_success_threshold_from_usd(object_name: str) -> float:
    """Return radians from the drive target to 80% of the revolute joint upper limit."""
    import math

    joint = _find_lever_revolute_joint_prim(object_name)
    upper_limit_deg = float(joint.GetAttribute("physics:upperLimit").Get())
    target_position_attr = joint.GetAttribute("drive:angular:physics:targetPosition")
    if target_position_attr and target_position_attr.Get() is not None:
        target_position_deg = float(target_position_attr.Get())
    else:
        target_position_deg = 0.0
    assert math.isfinite(
        upper_limit_deg
    ), f"{joint.GetPath()} needs a finite physics:upperLimit for success."
    threshold_rad = math.radians(abs(0.8 * upper_limit_deg - target_position_deg))
    return max(threshold_rad, math.radians(1.0))


def _quat_angle_from_rest(
    current_quat: torch.Tensor, rest_quat: torch.Tensor
) -> torch.Tensor:
    """Unsigned angular distance between two xyzw quaternions."""
    current_quat = current_quat / torch.linalg.norm(current_quat).clamp(min=1e-9)
    rest_quat = rest_quat / torch.linalg.norm(rest_quat).clamp(min=1e-9)
    dot = torch.sum(current_quat * rest_quat).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _lever_success_reached(
    env, object_name: str, rest_quat: torch.Tensor, threshold_rad: float
) -> bool:
    _, handle_quat = _object_pose(env, object_name)
    return bool(_quat_angle_from_rest(handle_quat, rest_quat) >= threshold_rad)


def _resolve_lever_eef_path() -> str | None:
    if args_cli.lever_eef_dataset_file is not None:
        return (
            None
            if args_cli.lever_eef_dataset_file.lower() == "none"
            else args_cli.lever_eef_dataset_file
        )
    root, _ = os.path.splitext(args_cli.dataset_file)
    return f"{root}_lever_eef.hdf5"


def main() -> None:
    env = _create_environment()
    assert (
        env.num_envs == 1
    ), f"Scripted lever recording only supports --num_envs 1, got {env.num_envs}"
    assert (
        env.action_manager.total_action_dim >= ALEX_ABILITY_HAND_WRIST_ACTION_DIM
    ), "This script targets the ability-hands (Pink IK, EE-pose action) embodiments."

    lever_eef_path = _resolve_lever_eef_path()
    lever_eef_file = (
        h5py.File(lever_eef_path, "w") if lever_eef_path is not None else None
    )

    recorded = 0
    with torch.inference_mode():
        while recorded < args_cli.num_demos and simulation_app.is_running():
            env.sim.reset()
            env.recorder_manager.reset()
            env.reset()

            lever_eef_states, lever_eef_actions = [], []
            success_threshold_rad = _lever_success_threshold_from_usd(
                args_cli.object_name
            )
            success_reached = False
            success_check_start_step = (
                args_cli.hold_steps
                + args_cli.close_steps
                + args_cli.approach_steps
                + args_cli.descend_steps
            )
            rest_handle_quat = None
            max_handle_angle_rad = torch.tensor(0.0, device=env.device)
            segments = _build_lever_path(env, env.device)
            for step_index, (pos, quat, hand) in enumerate(play_segments(segments)):
                targets = _bimanual_wrist_targets(env, args_cli.arm, pos, quat)
                if lever_eef_file is not None:
                    lever_eef_states.append(_lever_eef_state_vector(env))
                    lever_eef_actions.append(
                        _lever_eef_action_vector(env, targets, hand)
                    )
                action = _pink_ik_action(targets, hand).unsqueeze(0)
                env.step(action)
                if step_index >= success_check_start_step:
                    if rest_handle_quat is None:
                        _, rest_handle_quat = _object_pose(env, args_cli.object_name)
                    _, handle_quat = _object_pose(env, args_cli.object_name)
                    handle_angle_rad = _quat_angle_from_rest(
                        handle_quat, rest_handle_quat
                    )
                    max_handle_angle_rad = torch.maximum(
                        max_handle_angle_rad, handle_angle_rad
                    )
                    if handle_angle_rad >= success_threshold_rad:
                        success_reached = True
                        break

            if not success_reached:
                import math

                raise RuntimeError(
                    "Lever success threshold was not reached; not exporting this rollout as successful. "
                    f"Max handle motion was {math.degrees(float(max_handle_angle_rad)):.2f} deg; "
                    f"threshold is {math.degrees(float(success_threshold_rad)):.2f} deg."
                )
            export_episode_as_success(env)
            recorded = env.recorder_manager.exported_successful_episode_count
            print(
                f"Recorded {recorded}/{args_cli.num_demos} scripted lever demonstrations."
            )

            if lever_eef_file is not None:
                episode_group = lever_eef_file.create_group(f"data/demo_{recorded - 1}")
                episode_group.create_dataset(
                    "observation.state",
                    data=torch.stack(lever_eef_states).cpu().numpy(),
                )
                episode_group.create_dataset(
                    "action", data=torch.stack(lever_eef_actions).cpu().numpy()
                )

    env.close()
    if lever_eef_file is not None:
        lever_eef_file.close()
        print(
            f"test_obs_new-schema (48-dim state / 46-dim action) saved to: {lever_eef_path}"
        )
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
