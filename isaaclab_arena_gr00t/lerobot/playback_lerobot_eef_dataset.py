# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay LeRobot EEF episodes (H2Ozone/lever_eef) on Alex via Pink IK, with the
real ZED recordings shown in dockable Isaac windows.

The dataset stores gripper poses in the real robot's world frame using IHMC's hand
control frames. Calibrated constants (solved against FK of the joint-space twin
H2Ozone/lever_fingers) map each pose into a sim-world ``*_GRIPPER_Z_LINK`` target
consumed by the ``alex_v2_ability_hands`` Pink IK action. Neck joints (not part of
the 34-dim action) are written kinematically.

Usage (inside Docker, from repo root; needs ihmc-alex-sdk mounted via ``-m``)::

    /isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/playback_lerobot_eef_dataset.py \\
        --dataset_path datasets/lever_eef \\
        --select_episodes 0,1,2 \\
        alex_teleop_sandbox \\
        --embodiment alex_v2_ability_hands

Pass ``--headless`` for a tracking-error check without a GUI (dataset-camera
windows are skipped). ``--source action`` replays commanded instead of measured poses.

To sanity-check a trained GR00T checkpoint without a task environment, start a
GR00T server for the checkpoint and pass ``--policy_mode gr00t_remote``. The
script will feed the recorded ZED frames to the policy, convert predicted
dataset-frame EEF actions into the sim Pink IK action frame, and move Alex in
the sandbox while the old episode videos play alongside it.
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--dataset_path",
    type=str,
    default="datasets/lever_eef",
    help="LeRobot v3 dataset root.",
)
parser.add_argument(
    "--select_episodes",
    type=lambda arg: [int(part) for part in arg.split(",")],
    default=[],
    help="Comma-separated episode indices to play. Empty plays all episodes in order.",
)
parser.add_argument(
    "--source",
    type=str,
    choices=["state", "action"],
    default="state",
    help="Replay measured poses (default) or commanded action targets.",
)
parser.add_argument(
    "--playback_speed", type=float, default=1.0, help="Speed multiplier."
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=0,
    help="Limit frames per episode for smoke tests; 0 plays all.",
)
parser.add_argument(
    "--loop",
    action="store_true",
    default=False,
    help="Restart from the first episode when done.",
)
parser.add_argument(
    "--policy_mode",
    type=str,
    choices=["dataset_replay", "gr00t_remote"],
    default="dataset_replay",
    help="Replay dataset actions directly, or query a remote GR00T policy from recorded video frames.",
)
parser.add_argument(
    "--policy_config_yaml_path",
    type=str,
    default="isaaclab_arena_gr00t/policy/config/alex_lever_eef_gr00t_closedloop_config.yaml",
    help="Closed-loop policy config used for GR00T observation/action translation.",
)
parser.add_argument(
    "--remote_host", type=str, default="localhost", help="GR00T policy server hostname."
)
parser.add_argument(
    "--remote_port", type=int, default=5555, help="GR00T policy server port."
)
parser.add_argument(
    "--remote_api_token",
    type=str,
    default=None,
    help="Optional GR00T policy server API token.",
)
parser.add_argument(
    "--viz_policy_targets",
    action="store_true",
    default=False,
    help="Show blue/orange markers at the policy's predicted wrist targets.",
)
parser.add_argument(
    "--no_dataset_cameras",
    action="store_true",
    default=False,
    help="Do not open windows showing the recorded ZED videos.",
)
add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import json
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# region agent log
def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """Append a single NDJSON debug log line for this debug session."""
    try:
        import json
        import time

        entry = {
            "sessionId": "228c09",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("/home/bpratt/IsaacLab-Arena/.cursor/debug-228c09.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Logging must never break the main control flow.
        pass


# endregion

SOURCE_FEATURE_KEYS = {"state": "observation.state", "action": "action"}
EEF_ACTION_DIM = 34

# --- Calibrated frame constants -----------------------------------------------------
# Solved from H2Ozone/lever_fingers FK vs H2Ozone/lever_eef pose pairs (same 29 takes),
# with the dataset->sim world rotation constrained to pure yaw (both frames gravity-
# aligned). Residuals: left ~13 mm / right ~29 mm mean (right-hand excess is real
# pelvis sway during the lever pull; the sim base is fixed).
#
# Replay target (sim world) = T_pelvis_now * T_PELVIS_CALIB^-1 * A_INV * D * B_INV_<hand>
# where D is the dataset pose. The pelvis composition keeps the calibration valid if an
# env spawns Alex somewhere other than where the constants were solved.
_A_INV = ((0.043138, -0.482041, 0.028456), (0.0, 0.0, 0.0, 1.0))
_B_INV = {
    "left": (
        (0.013252, 0.007675, -0.002249),
        (-0.161324, -0.324033, -0.890589, 0.275369),
    ),
    "right": (
        (-0.073235, -0.007256, -0.00897),
        (-0.178244, -0.374856, -0.909779, 0.003727),
    ),
}
# Pelvis world pose (pos, quat xyzw) in the env the constants were solved in
# (alex_teleop_sandbox 0-yaw spawn: identity orientation, raw body_quat_w convention).
_PELVIS_CALIB = ((-0.4, -0.48682, 0.94296), (0.0, 0.0, 0.0, 1.0))

_NECK_MOTORS = ("neck_z", "neck_y")
_LEFT_HAND_MOTORS = (
    "left_ability_hand_index_q1",
    "left_ability_hand_index_q2",
    "left_ability_hand_middle_q1",
    "left_ability_hand_middle_q2",
    "left_ability_hand_ring_q1",
    "left_ability_hand_ring_q2",
    "left_ability_hand_pinky_q1",
    "left_ability_hand_pinky_q2",
    "left_ability_hand_thumb_q1",
    "left_ability_hand_thumb_q2",
)
_RIGHT_HAND_MOTORS = tuple(
    name.replace("left_", "right_") for name in _LEFT_HAND_MOTORS
)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of scalar-last (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    qc = np.array([-q[0], -q[1], -q[2], q[3]])
    return _quat_mul(_quat_mul(q, qv), qc)[:3]


def _pose_mul(a: tuple, b: tuple) -> tuple:
    """Compose two (pos, quat_xyzw) poses: a ∘ b."""
    pa, qa = np.asarray(a[0], dtype=np.float64), np.asarray(a[1], dtype=np.float64)
    pb, qb = np.asarray(b[0], dtype=np.float64), np.asarray(b[1], dtype=np.float64)
    return (pa + _quat_rotate(qa, pb), _quat_mul(qa, qb))


def _pose_inv(a: tuple) -> tuple:
    p, q = np.asarray(a[0], dtype=np.float64), np.asarray(a[1], dtype=np.float64)
    qc = np.array([-q[0], -q[1], -q[2], q[3]])
    return (-_quat_rotate(qc, p), qc)


def _quat_angle_deg_xyzw(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest angular distance between scalar-last quaternions."""
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa = qa / np.linalg.norm(qa)
    qb = qb / np.linalg.norm(qb)
    dot = abs(float(np.dot(qa, qb)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        import warp as wp

        if not isinstance(value, np.ndarray):
            value = wp.to_torch(value)
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _obs_from_result(result):
    return result[0] if isinstance(result, tuple) else result


def load_episodes(
    dataset_root: Path, feature_key: str
) -> tuple[dict[int, np.ndarray], list[str], float]:
    """Load episode tracks, motor names, and fps from a LeRobot v3 dataset."""
    with open(dataset_root / "meta" / "info.json") as f:
        info = json.load(f)
    motor_names = info["features"][feature_key]["names"]["motors"]
    fps = float(info["fps"])
    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    assert parquet_paths, f"No parquet files under {dataset_root / 'data'}"
    frames = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    episodes = {}
    for episode_index, ep_frames in frames.groupby("episode_index"):
        episodes[int(episode_index)] = np.stack(
            ep_frames.sort_values("frame_index")[feature_key].to_numpy()
        ).astype(np.float32)
    return episodes, motor_names, fps


class DatasetVideoReader:
    """Synchronized reader for recorded dataset videos, optionally mirrored into Isaac UI windows."""

    def __init__(self, dataset_root: Path, show_ui: bool):
        import cv2

        self._cv2 = cv2
        with open(dataset_root / "meta" / "info.json") as f:
            info = json.load(f)
        self._video_keys = [
            k for k, feat in info["features"].items() if feat.get("dtype") == "video"
        ]
        ep_paths = sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))
        self._episodes_meta = pd.concat(
            [pd.read_parquet(p) for p in ep_paths], ignore_index=True
        ).set_index("episode_index")
        self._root = dataset_root
        self._video_path_tpl = info["video_path"]
        self._captures: dict[str, object] = {}
        self._providers, self._windows = {}, {}
        if show_ui:
            import omni.ui as ui

            for key in self._video_keys:
                provider = ui.ByteImageProvider()
                window = ui.Window(
                    f"Dataset {key.split('.')[-1]}", width=480, height=380
                )
                with window.frame:
                    ui.ImageWithProvider(provider)
                self._providers[key] = provider
                self._windows[key] = window

    def start_episode(self, episode_index: int) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures = {}
        meta = self._episodes_meta.loc[episode_index]
        for key in self._video_keys:
            path = self._root / self._video_path_tpl.format(
                video_key=key,
                chunk_index=int(meta[f"videos/{key}/chunk_index"]),
                file_index=int(meta[f"videos/{key}/file_index"]),
            )
            cap = self._cv2.VideoCapture(str(path))
            cap.set(
                self._cv2.CAP_PROP_POS_MSEC,
                float(meta[f"videos/{key}/from_timestamp"]) * 1000.0,
            )
            self._captures[key] = cap

    def read_frame(self) -> dict[str, np.ndarray]:
        frames = {}
        for key, cap in self._captures.items():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read recorded video frame for {key}")
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            frames[key] = rgb
            if key not in self._providers:
                continue
            rgba = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGBA)
            height, width = rgba.shape[:2]
            self._providers[key].set_bytes_data(
                rgba.flatten().tolist(), [width, height]
            )
        return frames

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()


@dataclass
class DatasetLayout:
    motor_index: dict[str, int]
    pose_slices: dict[str, list[int]]
    hand_indices: list[int]
    left_hand_indices: list[int]
    right_hand_indices: list[int]
    neck_indices: list[int]


@dataclass
class SimHandles:
    robot: object
    pelvis_id: int
    left_id: int
    right_id: int
    neck_ids: torch.Tensor


@dataclass
class Calibration:
    world_from_dataset: tuple

    def dataset_pose_to_action(
        self, frame: np.ndarray, pose_slices: dict[str, list[int]], hand: str
    ) -> np.ndarray:
        raw = frame[pose_slices[hand]].astype(np.float64)
        pose = _pose_mul(
            _pose_mul(self.world_from_dataset, (raw[:3], raw[3:7])), _B_INV[hand]
        )
        quat_xyzw = pose[1] / np.linalg.norm(pose[1])
        # Pink IK / Isaac Lab matrix_from_quat expect scalar-last (x, y, z, w).
        return np.concatenate([pose[0], quat_xyzw])

    def sim_pose_to_dataset_pose(
        self, sim_pose_xyzw: np.ndarray, hand: str
    ) -> np.ndarray:
        # Isaac Lab 3.0 body_quat_w is already scalar-last (x, y, z, w).
        sim_pose = (sim_pose_xyzw[:3], sim_pose_xyzw[3:7])
        dataset_pose = _pose_mul(
            _pose_mul(_pose_inv(self.world_from_dataset), sim_pose),
            _pose_inv(_B_INV[hand]),
        )
        quat_xyzw = dataset_pose[1] / np.linalg.norm(dataset_pose[1])
        return np.concatenate([dataset_pose[0], quat_xyzw]).astype(np.float32)


@dataclass
class PolicyReplayContext:
    client: object
    policy_config: object
    policy_joints_config: dict
    robot_state_joints_config: dict
    modality_configs: dict
    task_description: str


def build_dataset_layout(
    motor_names: list[str], hand_slot_names: list[str]
) -> DatasetLayout:
    motor_index = {name: i for i, name in enumerate(motor_names)}
    pose_slices = {
        "left": [
            motor_index[f"left_gripper_{c}"]
            for c in ("x", "y", "z", "qx", "qy", "qz", "qs")
        ],
        "right": [
            motor_index[f"right_gripper_{c}"]
            for c in ("x", "y", "z", "qx", "qy", "qz", "qs")
        ],
    }
    return DatasetLayout(
        motor_index=motor_index,
        pose_slices=pose_slices,
        hand_indices=[motor_index[name] for name in hand_slot_names],
        left_hand_indices=[motor_index[name] for name in _LEFT_HAND_MOTORS],
        right_hand_indices=[motor_index[name] for name in _RIGHT_HAND_MOTORS],
        neck_indices=[motor_index[name] for name in _NECK_MOTORS],
    )


def build_calibration(robot, pelvis_id: int) -> Calibration:
    pelvis_p = _to_numpy(robot.data.body_pos_w)[0, pelvis_id].astype(np.float64)
    pelvis_q_xyzw = _to_numpy(robot.data.body_quat_w)[0, pelvis_id].astype(np.float64)
    pelvis_now = (pelvis_p, pelvis_q_xyzw)
    world_from_dataset = _pose_mul(
        _pose_mul(pelvis_now, _pose_inv(_PELVIS_CALIB)), _A_INV
    )
    return Calibration(world_from_dataset=world_from_dataset)


def build_dataset_replay_action(
    frame: np.ndarray, layout: DatasetLayout, calibration: Calibration
) -> np.ndarray:
    action = np.zeros(EEF_ACTION_DIM, dtype=np.float32)
    action[0:7] = calibration.dataset_pose_to_action(frame, layout.pose_slices, "left")
    action[7:14] = calibration.dataset_pose_to_action(
        frame, layout.pose_slices, "right"
    )
    action[14:34] = frame[layout.hand_indices]
    return action


def select_first_policy_step(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        return arr[0, 0]
    if arr.ndim == 2:
        return arr[0]
    return arr


def policy_action_to_dataset_frame(
    policy_action: dict, layout: DatasetLayout
) -> np.ndarray:
    frame = np.zeros(len(layout.motor_index), dtype=np.float32)
    frame[layout.pose_slices["left"]] = select_first_policy_step(
        policy_action["left_wrist_pose"]
    )
    frame[layout.pose_slices["right"]] = select_first_policy_step(
        policy_action["right_wrist_pose"]
    )
    frame[layout.left_hand_indices] = select_first_policy_step(
        policy_action["left_hand"]
    )
    frame[layout.right_hand_indices] = select_first_policy_step(
        policy_action["right_hand"]
    )
    if "neck" in policy_action:
        frame[layout.neck_indices] = select_first_policy_step(policy_action["neck"])
    return frame


def get_body_pose_xyzw(robot, body_id: int, env_origin: np.ndarray) -> np.ndarray:
    pos = _to_numpy(robot.data.body_pos_w)[0, body_id].astype(np.float64) - env_origin
    quat_xyzw = _to_numpy(robot.data.body_quat_w)[0, body_id].astype(np.float64)
    return np.concatenate([pos, quat_xyzw])


def build_dataset_state_from_sim(
    obs: dict, env, handles: SimHandles, layout: DatasetLayout, calibration: Calibration
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    joint_pos = _to_numpy(obs["policy"]["robot_joint_pos"])
    env_origin = _to_numpy(env.scene.env_origins)[0].astype(np.float64)
    left_pose = get_body_pose_xyzw(handles.robot, handles.left_id, env_origin)
    right_pose = get_body_pose_xyzw(handles.robot, handles.right_id, env_origin)
    eef_pose_policy = {
        "left_wrist_pose": calibration.sim_pose_to_dataset_pose(left_pose, "left")[
            None, :
        ],
        "right_wrist_pose": calibration.sim_pose_to_dataset_pose(right_pose, "right")[
            None, :
        ],
    }
    return joint_pos, eef_pose_policy


def dataset_video_frames_for_policy(
    frames: dict[str, np.ndarray], modality_configs: dict
) -> list[np.ndarray]:
    rgb_list = []
    for video_key in modality_configs["video"].modality_keys:
        original_key = f"observation.images.{video_key}"
        if original_key not in frames:
            raise KeyError(
                f"Recorded video {original_key} is missing; available videos: {list(frames)}"
            )
        rgb_list.append(frames[original_key][None, ...])
    return rgb_list


def create_policy_replay_context() -> PolicyReplayContext:
    from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

    from isaaclab_arena_gr00t.policy.gr00t_core import load_gr00t_joint_configs
    from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
        Gr00tClosedloopPolicyConfig,
    )
    from isaaclab_arena_gr00t.utils.io_utils import (
        create_config_from_yaml,
        load_gr00t_modality_config_from_file,
    )

    policy_config = create_config_from_yaml(
        args_cli.policy_config_yaml_path, Gr00tClosedloopPolicyConfig
    )
    policy_joints_config, _robot_action_joints_config, robot_state_joints_config = (
        load_gr00t_joint_configs(policy_config)
    )
    modality_configs = load_gr00t_modality_config_from_file(
        policy_config.modality_config_path,
        policy_config.embodiment_tag,
    )
    client = Gr00tPolicyClient(
        host=args_cli.remote_host,
        port=args_cli.remote_port,
        api_token=args_cli.remote_api_token,
        strict=False,
    )
    if not client.ping():
        raise ConnectionError(
            f"Cannot reach GR00T policy server at {args_cli.remote_host}:{args_cli.remote_port}"
        )
    return PolicyReplayContext(
        client=client,
        policy_config=policy_config,
        policy_joints_config=policy_joints_config,
        robot_state_joints_config=robot_state_joints_config,
        modality_configs=modality_configs,
        task_description=policy_config.language_instruction,
    )


def query_gr00t_policy(
    context: PolicyReplayContext,
    recorded_frames: dict[str, np.ndarray],
    obs: dict,
    env,
    handles: SimHandles,
    layout: DatasetLayout,
    calibration: Calibration,
) -> dict:
    from isaaclab_arena_gr00t.policy.gr00t_core import build_gr00t_policy_observations

    joint_pos_sim_np, eef_pose_policy = build_dataset_state_from_sim(
        obs, env, handles, layout, calibration
    )
    policy_observation = build_gr00t_policy_observations(
        rgb_list_np=dataset_video_frames_for_policy(
            recorded_frames, context.modality_configs
        ),
        joint_pos_sim_np=joint_pos_sim_np,
        task_description=context.task_description,
        policy_config=context.policy_config,
        robot_state_joints_config=context.robot_state_joints_config,
        policy_joints_config=context.policy_joints_config,
        modality_configs=context.modality_configs,
        eef_pose_policy=eef_pose_policy,
    )
    policy_action, _info = context.client.get_action(policy_observation)
    return policy_action


def write_neck_targets(
    frame: np.ndarray, robot, neck_ids: torch.Tensor, layout: DatasetLayout, device: str
) -> None:
    neck = torch.as_tensor(
        frame[layout.neck_indices], dtype=torch.float32, device=device
    ).unsqueeze(0)
    robot.write_joint_position_to_sim_index(position=neck, joint_ids=neck_ids)
    robot.set_joint_position_target_index(target=neck, joint_ids=neck_ids)


def target_errors(
    achieved_pose_xyzw: np.ndarray, target_action_pose_xyzw: np.ndarray
) -> tuple[float, float]:
    pos_err = float(
        np.linalg.norm(achieved_pose_xyzw[:3] - target_action_pose_xyzw[:3])
    )
    return pos_err, _quat_angle_deg_xyzw(
        achieved_pose_xyzw[3:7], target_action_pose_xyzw[3:7]
    )


def demo_delta_metrics(
    predicted_frame: np.ndarray, demo_frame: np.ndarray, layout: DatasetLayout
) -> dict[str, float]:
    metrics = {}
    for hand in ("left", "right"):
        idx = layout.pose_slices[hand]
        metrics[f"{hand}_pos_delta"] = float(
            np.linalg.norm(predicted_frame[idx[:3]] - demo_frame[idx[:3]])
        )
        metrics[f"{hand}_rot_delta"] = _quat_angle_deg_xyzw(
            predicted_frame[idx[3:7]], demo_frame[idx[3:7]]
        )
    return metrics


def main():
    from isaaclab_arena.embodiments.alex.alex import (
        ABILITY_HAND_JOINT_NAMES_LIST,
        ABILITY_HAND_TELEOP_JOINT_ORDER,
    )

    dataset_root = Path(args_cli.dataset_path)
    feature_key = SOURCE_FEATURE_KEYS[args_cli.source]
    episodes, motor_names, fps = load_episodes(dataset_root, feature_key)
    print(
        f"Loaded {len(episodes)} episodes from {dataset_root} (source: {feature_key}, {fps} fps)"
    )
    demo_action_episodes, _action_motor_names, _ = load_episodes(dataset_root, "action")

    selected = args_cli.select_episodes or sorted(episodes)
    for episode_index in selected:
        assert episode_index in episodes, f"Episode {episode_index} not in dataset"

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    env = gym.make(env_name, cfg=env_cfg)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)
    env = env.unwrapped
    robot = env.scene["robot"]

    # The Pink IK action term resolves hand_joint_names WITHOUT preserve_order, so the
    # last 20 action slots follow the articulation's joint order, not the cfg list order.
    # Mirror that exact call so each dataset channel lands in the slot that drives it.
    _, hand_slot_names = robot.find_joints(ABILITY_HAND_TELEOP_JOINT_ORDER)
    layout = build_dataset_layout(motor_names, hand_slot_names)

    # region agent log
    # Hypotheses:
    # H1: Policy/right-hand pinky channels are near-zero even when other fingers close.
    # H2: Dataset -> action mapping scrambles or drops right-hand pinky channels.
    # H3: Sim joint positions for right-hand pinky do not track commanded action slots.
    _agent_debug_log(
        run_id="pre-run",
        hypothesis_id="H2",
        location="playback_lerobot_eef_dataset.main:layout",
        message="lever_eef layout indices",
        data={
            "motor_names_sample": motor_names[:48],
            "hand_indices": layout.hand_indices,
            "left_hand_indices": layout.left_hand_indices,
            "right_hand_indices": layout.right_hand_indices,
        },
    )
    # endregion

    neck_ids_list, neck_resolved = robot.find_joints(
        [m.upper() for m in _NECK_MOTORS], preserve_order=True
    )
    assert list(neck_resolved) == [m.upper() for m in _NECK_MOTORS]
    neck_ids = torch.tensor(neck_ids_list, dtype=torch.int32, device=env.device)

    pelvis_id, _ = robot.find_bodies(["PELVIS_LINK"])
    left_id, _ = robot.find_bodies(["LEFT_GRIPPER_Z_LINK"])
    right_id, _ = robot.find_bodies(["RIGHT_GRIPPER_Z_LINK"])
    handles = SimHandles(
        robot=robot,
        pelvis_id=int(pelvis_id[0]),
        left_id=int(left_id[0]),
        right_id=int(right_id[0]),
        neck_ids=neck_ids,
    )

    steps_per_frame_f = (1.0 / fps) / env.step_dt / args_cli.playback_speed
    action_dim = env.action_manager.total_action_dim
    assert (
        action_dim == EEF_ACTION_DIM
    ), f"Expected the {EEF_ACTION_DIM}-dim ability-hand Pink IK action space, got {action_dim}"

    policy_context = (
        create_policy_replay_context()
        if args_cli.policy_mode == "gr00t_remote"
        else None
    )
    video_reader = None
    if args_cli.policy_mode == "gr00t_remote" or (
        not args_cli.no_dataset_cameras and not args_cli.headless
    ):
        video_reader = DatasetVideoReader(
            dataset_root=dataset_root,
            show_ui=(not args_cli.no_dataset_cameras and not args_cli.headless),
        )

    action_target_marker = None
    if args_cli.viz_policy_targets:
        from isaaclab_arena.evaluation.action_target_marker import ActionTargetMarker

        action_target_marker = ActionTargetMarker()
        print("Policy target markers enabled (blue=left, orange=right)")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            for episode_index in selected:
                track = episodes[episode_index]
                if args_cli.max_frames > 0:
                    track = track[: args_cli.max_frames]
                demo_action_track = demo_action_episodes[episode_index]
                print(
                    f"Playing episode {episode_index} ({len(track)} frames, {len(track) / fps:.1f}s)",
                    flush=True,
                )
                obs = _obs_from_result(env.reset())
                calibration = build_calibration(robot, handles.pelvis_id)
                if video_reader is not None:
                    video_reader.start_episode(episode_index)
                if policy_context is not None:
                    policy_context.client.reset()
                step_debt = 0.0
                pos_errs, rot_errs = [], []
                demo_pos_deltas, demo_rot_deltas, action_jumps = [], [], []
                previous_policy_action = None
            for frame_index, frame in enumerate(track):
                    recorded_frames = (
                        video_reader.read_frame() if video_reader is not None else None
                    )
                    if policy_context is None:
                        action = build_dataset_replay_action(frame, layout, calibration)
                        predicted_frame = None
                    else:
                        policy_action = query_gr00t_policy(
                            policy_context,
                            recorded_frames,
                            obs,
                            env,
                            handles,
                            layout,
                            calibration,
                        )
                        predicted_frame = policy_action_to_dataset_frame(
                            policy_action, layout
                        )
                        action = build_dataset_replay_action(
                            predicted_frame, layout, calibration
                        )
                        deltas = demo_delta_metrics(
                            predicted_frame, demo_action_track[frame_index], layout
                        )
                        demo_pos_deltas.extend(
                            [deltas["left_pos_delta"], deltas["right_pos_delta"]]
                        )
                        demo_rot_deltas.extend(
                            [deltas["left_rot_delta"], deltas["right_rot_delta"]]
                        )
                        if previous_policy_action is not None:
                            action_jumps.append(
                                float(np.linalg.norm(action - previous_policy_action))
                            )
                        previous_policy_action = action.copy()

                    # region agent log
                    if frame_index < 5:
                        # Log dataset and policy pinky channels for early frames.
                        right_pinky_q1_name = "right_ability_hand_pinky_q1"
                        right_pinky_q2_name = "right_ability_hand_pinky_q2"
                        right_pinky_dataset = {}
                        for name in (right_pinky_q1_name, right_pinky_q2_name):
                            idx = layout.motor_index.get(name)
                            right_pinky_dataset[name] = float(frame[idx]) if idx is not None else None

                        right_pinky_policy = {}
                        if predicted_frame is not None:
                            try:
                                right_indices = layout.right_hand_indices
                                right_vals = predicted_frame[right_indices]
                                names = [n for n in ABILITY_HAND_JOINT_NAMES_LIST if n.startswith("right_")]
                                right_pinky_policy = {
                                    "right_hand_vec": right_vals.tolist(),
                                    "names": names,
                                }
                            except Exception:
                                right_pinky_policy = {"error": "could not map right_hand_indices"}

                        # Action hand block indices 14:34 (20 dims).
                        hand_block = action[14:34].tolist()
                        _agent_debug_log(
                            run_id="pre-finger-step",
                            hypothesis_id="H1",
                            location="playback_lerobot_eef_dataset.main:frame_loop",
                            message="lever_eef right-hand pinky channels",
                            data={
                                "episode_index": int(episode_index),
                                "frame_index": int(frame_index),
                                "right_pinky_dataset": right_pinky_dataset,
                                "right_pinky_policy": right_pinky_policy,
                                "hand_block_14_34": hand_block,
                            },
                        )
                    # endregion

                    action_t = torch.as_tensor(
                        action, dtype=torch.float32, device=env.device
                    ).unsqueeze(0)

                    write_neck_targets(
                        predicted_frame if predicted_frame is not None else frame,
                        robot,
                        neck_ids,
                        layout,
                        env.device,
                    )

                    step_debt += steps_per_frame_f
                    while step_debt >= 1.0:
                        obs = _obs_from_result(env.step(action_t))
                        step_debt -= 1.0
                    if action_target_marker is not None:
                        action_target_marker.update(action_t)

                    env_origin = _to_numpy(env.scene.env_origins)[0].astype(np.float64)
                    achieved_right = get_body_pose_xyzw(
                        robot, handles.right_id, env_origin
                    )
                    pos_err, rot_err = target_errors(achieved_right, action[7:14])
                    pos_errs.append(pos_err)
                    rot_errs.append(rot_err)

                    # region agent log
                    if frame_index < 5:
                        try:
                            # Sample right-hand finger joint positions after stepping to see if pinky tracks others.
                            joint_names = ABILITY_HAND_JOINT_NAMES_LIST
                            joint_ids, resolved = robot.find_joints(joint_names, preserve_order=True)
                            joint_pos = _to_numpy(robot.data.joint_pos)[0]
                            right_joint_snapshot = {
                                name: float(joint_pos[jid])
                                for name, jid in zip(resolved, joint_ids)
                                if name.startswith("right_ability_hand_")
                            }
                        except Exception:
                            right_joint_snapshot = {"error": "could not read right-hand joint_pos"}

                        _agent_debug_log(
                            run_id="post-step",
                            hypothesis_id="H3",
                            location="playback_lerobot_eef_dataset.main:post_step",
                            message="lever_eef right-hand finger joint_pos snapshot",
                            data={
                                "episode_index": int(episode_index),
                                "frame_index": int(frame_index),
                                "right_joint_snapshot": right_joint_snapshot,
                            },
                        )
                    # endregion
                    if not simulation_app.is_running() or simulation_app.is_exiting():
                        break
                    # IK convergence + calibration + base-sway error, right hand (the working arm).
                    summary = (
                        f"Episode {episode_index} done. Right-hand target error: "
                        f"mean {np.mean(pos_errs)*1000:.0f} mm pos / {np.mean(rot_errs):.1f} deg rot, "
                        f"max {np.max(pos_errs)*1000:.0f} mm / {np.max(rot_errs):.1f} deg"
                    )
                    if policy_context is not None and demo_pos_deltas:
                        summary += (
                            f"; policy-vs-demo wrist target delta: mean {np.mean(demo_pos_deltas)*1000:.0f} mm /"
                            f" {np.mean(demo_rot_deltas):.1f} deg"
                        )
                        if action_jumps:
                            summary += f"; mean action jump {np.mean(action_jumps):.3f}"
                    print(summary, flush=True)
            if not args_cli.loop:
                break

    if video_reader is not None:
        video_reader.close()
    print("Playback finished.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
