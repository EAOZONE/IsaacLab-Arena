# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record Alex + lever demonstrations from a scripted Cartesian path, not teleop.

The lever's live world pose is read straight out of the scene (no human
operator needed). The scripted motion closes the hand into a fist, moves above
the lever, then pushes straight down through the lever's range -- rather than
grasping the handle and rotating the wrist to match it -- since a fixed-fist
push is an easier target for a policy to learn than a precision grasp. The
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
packed into the same 36-dim layout as the real-hardware
`H2Ozone/lever_eef <https://huggingface.co/datasets/H2Ozone/lever_eef>`_
dataset (see its ``meta/info.json`` and
``isaaclab_arena_gr00t/embodiments/alex/alex_lever_eef_modality.json``):
``[left_wrist_pose(7), right_wrist_pose(7), left_hand(10), right_hand(10),
neck(2)]``, hands grouped per finger (q1, q2) rather than the Pink IK action
term's interleaved order. This script doesn't drive the neck (the
ability-hands embodiment doesn't expose it as an action), so the neck columns
are the robot's actual (constant) neck joint reading, not a scripted motion.
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
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", type=str, required=True, help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument("--num_demos", type=int, default=1, help="Number of demonstrations to record.")
parser.add_argument(
    "--object_name",
    type=str,
    default="lever_revolute",
    help="Scene key of the lever rigid object to reach for (default matches alex_empty's"
    " Lever_revolute.usd asset name).",
)
parser.add_argument("--arm", type=str, choices=["left", "right"], default="right", help="Which arm pushes the lever.")
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
    help="How far to press the lever through its 0-90 deg range (stays clear of the hard limit). Used"
    " only to size the vertical push depth via the hinge geometry -- the hand still moves in a"
    " straight line down (see --min_push_depth), not along the lever's own arc.",
)
parser.add_argument(
    "--min_push_depth",
    type=float,
    default=0.03,
    help="Floor [m] on the world-Z push depth computed from --push_target_deg, in case the hinge-geometry"
    " estimate comes out too small (or the wrong sign) to actually depress the lever.",
)
parser.add_argument("--close_fraction", type=float, default=1.0, help="How far to close the hand into a fist.")
parser.add_argument("--hold_steps", type=int, default=15, help="Steps to settle at the arm's starting pose.")
parser.add_argument("--close_steps", type=int, default=20, help="Steps to close the hand into a fist at the home pose.")
parser.add_argument("--approach_steps", type=int, default=45, help="Steps from home to the standoff above the lever.")
parser.add_argument(
    "--push_steps", type=int, default=90, help="Steps to press from the standoff down through --push_target_deg."
)
parser.add_argument("--dwell_steps", type=int, default=15, help="Steps to hold the pressed-down pose.")
parser.add_argument("--retreat_steps", type=int, default=30, help="Steps back up to the standoff above the lever.")
parser.add_argument("--release_steps", type=int, default=20, help="Steps to open the hand at the standoff.")
parser.add_argument("--return_steps", type=int, default=30, help="Steps back to the arm's starting pose.")
parser.add_argument(
    "--lever_eef_dataset_file",
    type=str,
    default=None,
    help="Also write observation.state/action arrays matching the H2Ozone/lever_eef 36-dim schema to"
    " this hdf5. Defaults to --dataset_file with a '_lever_eef' suffix; pass 'none' to skip.",
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
)
from isaaclab_arena.utils.cartesian_waypoints import LinearSegment, play_segments
from isaaclab_arena.utils.isaaclab_utils.recorders import ArenaEnvRecorderManagerCfg

# H2Ozone/lever_eef's hand-joint layout groups per finger (q1, q2) and per side, unlike the
# Pink IK action term's interleaved ABILITY_HAND_TELEOP_JOINT_ORDER. See that dataset's
# meta/info.json and isaaclab_arena_gr00t/embodiments/alex/alex_lever_eef_modality.json.
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
_PINK_IK_TO_LEVER_EEF_HAND_PERM = [ABILITY_HAND_TELEOP_JOINT_ORDER.index(name) for name in _LEVER_EEF_HAND_JOINT_ORDER]
_LEVER_EEF_NECK_JOINT_NAMES = ["NECK_Z", "NECK_Y"]


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


def _wrist_pose(env, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live (pos, quat_xyzw) of ``link_name`` on the robot, relative to its env origin."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies([link_name])
    idx = int(body_ids[0])
    pos = wp.to_torch(robot.data.body_pos_w)[0, idx] - env.scene.env_origins[0]
    quat = wp.to_torch(robot.data.body_quat_w)[0, idx]
    return pos.clone(), quat.clone()


def _object_pose(env, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live (pos, quat_xyzw) of the lever rigid object, relative to its env origin."""
    assert object_name in env.scene.keys(), (
        f"'{object_name}' not found in the scene (available: {list(env.scene.keys())}). Pass --object_name to match"
        " the --usd asset's prim name."
    )
    data = env.scene[object_name].data
    pos = wp.to_torch(data.root_pos_w)[0] - env.scene.env_origins[0]
    quat = wp.to_torch(data.root_quat_w)[0]
    return pos.clone(), quat.clone()


def _build_lever_path(env, device: torch.device):
    """Close hand -> move above the lever -> push straight down -> dwell -> retreat -> release -> home.

    Both wrist orientation and the vertical (approach/press/retreat) motion are
    deliberately decoupled from the lever's own orientation: the fist keeps its
    starting (home) orientation the whole time -- pressing down doesn't need any
    wrist rotation -- and "up"/"down" are literal world Z, not the handle's local
    frame (which is tilted/yawed, especially with ``--lever_dr``; rotating those
    offsets through it was sending the hand sideways instead of down). Only the
    horizontal (x, y) placement of the contact point tracks the lever's live
    pose, since that's what determines *where* on the lever to press.
    """
    import math

    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    gripper_link = f"{args_cli.arm.upper()}_GRIPPER_Z_LINK"
    home_pos, home_quat = _wrist_pose(env, gripper_link)
    handle_pos, handle_quat = _object_pose(env, args_cli.object_name)

    push_local_offset = torch.tensor(args_cli.push_local_offset, device=device, dtype=home_pos.dtype)
    push_wrist_rot_offset = torch.tensor(args_cli.push_wrist_rot_offset, device=device, dtype=home_quat.dtype)
    push_quat = quat_mul(home_quat.unsqueeze(0), push_wrist_rot_offset.unsqueeze(0)).squeeze(0)

    contact_pos = handle_pos + quat_apply(handle_quat.unsqueeze(0), push_local_offset.unsqueeze(0)).squeeze(0)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=contact_pos.dtype)
    above_pos = contact_pos + args_cli.approach_height * world_up

    # Depth estimate only: how far the contact point would drop in world Z if the lever rotated
    # through --push_target_deg about its hinge. The hand still travels in a straight vertical
    # line from contact_pos, not along this arc -- contact dynamics do the rest.
    axis_world = quat_apply(handle_quat.unsqueeze(0), torch.tensor([[0.0, 1.0, 0.0]], device=device)).squeeze(0)
    push_angle = torch.tensor([math.radians(args_cli.push_target_deg)], device=device, dtype=axis_world.dtype)
    push_rot = quat_from_angle_axis(push_angle, axis_world.unsqueeze(0)).squeeze(0)
    rotated_contact_pos = handle_pos + quat_apply(
        push_rot.unsqueeze(0), (contact_pos - handle_pos).unsqueeze(0)
    ).squeeze(0)
    push_depth = max(float(contact_pos[2] - rotated_contact_pos[2]), args_cli.min_push_depth)
    pushed_pos = contact_pos - push_depth * world_up

    left_close = args_cli.close_fraction if args_cli.arm == "left" else 0.0
    right_close = args_cli.close_fraction if args_cli.arm == "right" else 0.0
    open_hand = build_ability_hand_joint_action(0.0, 0.0, device=device)
    closed_hand = build_ability_hand_joint_action(left_close, right_close, device=device)

    return [
        LinearSegment(home_pos, home_quat, open_hand, home_pos, home_quat, open_hand, args_cli.hold_steps),
        LinearSegment(home_pos, home_quat, open_hand, home_pos, home_quat, closed_hand, args_cli.close_steps),
        LinearSegment(home_pos, home_quat, closed_hand, above_pos, push_quat, closed_hand, args_cli.approach_steps),
        LinearSegment(above_pos, push_quat, closed_hand, pushed_pos, push_quat, closed_hand, args_cli.push_steps),
        LinearSegment(pushed_pos, push_quat, closed_hand, pushed_pos, push_quat, closed_hand, args_cli.dwell_steps),
        LinearSegment(pushed_pos, push_quat, closed_hand, above_pos, push_quat, closed_hand, args_cli.retreat_steps),
        LinearSegment(above_pos, push_quat, closed_hand, above_pos, push_quat, open_hand, args_cli.release_steps),
        LinearSegment(above_pos, push_quat, open_hand, home_pos, home_quat, open_hand, args_cli.return_steps),
    ]


def _bimanual_wrist_targets(
    env, active_arm: str, pos: torch.Tensor, quat: torch.Tensor
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Wrist targets for both arms: ``active_arm`` gets (pos, quat); the other holds its current pose."""
    idle_arm = "left" if active_arm == "right" else "right"
    idle_pos, idle_quat = _wrist_pose(env, f"{idle_arm.upper()}_GRIPPER_Z_LINK")
    return {active_arm: (pos, quat), idle_arm: (idle_pos, idle_quat)}


def _pink_ik_action(targets: dict[str, tuple[torch.Tensor, torch.Tensor]], hand: torch.Tensor) -> torch.Tensor:
    """Assemble the 34-D ability-hand Pink IK action from bimanual wrist targets + hand joints."""
    left_pos, left_quat = targets["left"]
    right_pos, right_quat = targets["right"]
    return torch.cat([left_pos, left_quat, right_pos, right_quat, hand])


def _hand_joint_pos(env) -> torch.Tensor:
    """Live ability-hand joint positions, in ``ABILITY_HAND_TELEOP_JOINT_ORDER``."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(ABILITY_HAND_TELEOP_JOINT_ORDER, preserve_order=True)
    return wp.to_torch(robot.data.joint_pos)[0, joint_ids].clone()


def _neck_joint_pos(env) -> torch.Tensor:
    """Live (NECK_Z, NECK_Y) joint positions -- this script never commands the neck."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(_LEVER_EEF_NECK_JOINT_NAMES, preserve_order=True)
    return wp.to_torch(robot.data.joint_pos)[0, joint_ids].clone()


def _lever_eef_vector(
    left_pos: torch.Tensor,
    left_quat: torch.Tensor,
    right_pos: torch.Tensor,
    right_quat: torch.Tensor,
    hand_pink_ik_order: torch.Tensor,
    neck: torch.Tensor,
) -> torch.Tensor:
    """Pack into the H2Ozone/lever_eef 36-dim layout (wrist poses, grouped hand joints, neck)."""
    hand_grouped = hand_pink_ik_order[_PINK_IK_TO_LEVER_EEF_HAND_PERM]
    return torch.cat([left_pos, left_quat, right_pos, right_quat, hand_grouped, neck])


def export_episode_as_success(env) -> None:
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes([0])


def _resolve_lever_eef_path() -> str | None:
    if args_cli.lever_eef_dataset_file is not None:
        return None if args_cli.lever_eef_dataset_file.lower() == "none" else args_cli.lever_eef_dataset_file
    root, _ = os.path.splitext(args_cli.dataset_file)
    return f"{root}_lever_eef.hdf5"


def main() -> None:
    env = _create_environment()
    assert env.num_envs == 1, f"Scripted lever recording only supports --num_envs 1, got {env.num_envs}"
    assert (
        env.action_manager.total_action_dim >= ALEX_ABILITY_HAND_WRIST_ACTION_DIM
    ), "This script targets the ability-hands (Pink IK, EE-pose action) embodiments."

    lever_eef_path = _resolve_lever_eef_path()
    lever_eef_file = h5py.File(lever_eef_path, "w") if lever_eef_path is not None else None

    recorded = 0
    with torch.inference_mode():
        while recorded < args_cli.num_demos and simulation_app.is_running():
            env.sim.reset()
            env.recorder_manager.reset()
            env.reset()

            lever_eef_states, lever_eef_actions = [], []
            segments = _build_lever_path(env, env.device)
            for pos, quat, hand in play_segments(segments):
                targets = _bimanual_wrist_targets(env, args_cli.arm, pos, quat)
                if lever_eef_file is not None:
                    state_left = _wrist_pose(env, "LEFT_GRIPPER_Z_LINK")
                    state_right = _wrist_pose(env, "RIGHT_GRIPPER_Z_LINK")
                    neck = _neck_joint_pos(env)
                    lever_eef_states.append(_lever_eef_vector(*state_left, *state_right, _hand_joint_pos(env), neck))
                    lever_eef_actions.append(_lever_eef_vector(*targets["left"], *targets["right"], hand, neck))
                action = _pink_ik_action(targets, hand).unsqueeze(0)
                env.step(action)

            export_episode_as_success(env)
            recorded = env.recorder_manager.exported_successful_episode_count
            print(f"Recorded {recorded}/{args_cli.num_demos} scripted lever demonstrations.")

            if lever_eef_file is not None:
                episode_group = lever_eef_file.create_group(f"data/demo_{recorded - 1}")
                episode_group.create_dataset("observation.state", data=torch.stack(lever_eef_states).cpu().numpy())
                episode_group.create_dataset("action", data=torch.stack(lever_eef_actions).cpu().numpy())

    env.close()
    if lever_eef_file is not None:
        lever_eef_file.close()
        print(f"Lever_eef-schema (36-dim) state/action saved to: {lever_eef_path}")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
