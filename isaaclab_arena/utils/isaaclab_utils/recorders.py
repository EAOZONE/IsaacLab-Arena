# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
import torch
import warp as wp

from isaaclab_arena.embodiments.alex.alex import ABILITY_HAND_TELEOP_JOINT_ORDER

_TEST_OBS_NEW_STATE_DIM = 48
_TEST_OBS_NEW_ACTION_DIM = 46
_TEST_OBS_NEW_ARENA_ACTION_DIM = 34
_TEST_OBS_NEW_LEFT_FOREARM_LINK = "LEFT_WRIST_Z_LINK"
_TEST_OBS_NEW_RIGHT_FOREARM_LINK = "RIGHT_WRIST_Z_LINK"
_TEST_OBS_NEW_HEAD_LINK = "HEAD_LINK"
_TEST_OBS_NEW_SPINE_JOINTS = ("SPINE_Z", "SPINE_Y")
_TEST_OBS_NEW_GROUPED_HAND_NAMES = [
    f"{side}_ability_hand_{finger}_{joint}"
    for side in ("left", "right")
    for finger in ("index", "middle", "ring", "pinky", "thumb")
    for joint in ("q1", "q2")
]
_TEST_OBS_NEW_GROUPED_FROM_PINK = [
    ABILITY_HAND_TELEOP_JOINT_ORDER.index(name)
    for name in _TEST_OBS_NEW_GROUPED_HAND_NAMES
]


def _as_torch(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def _quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    # Isaac Lab 3.0's body_quat_w is already scalar-last xyzw in this stack.
    # Keeping this as a named helper makes the schema assumption explicit.
    return quat


def _body_pose(env, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies([body_name])
    idx = int(body_ids[0])
    pos = _as_torch(robot.data.body_pos_w)[:, idx] - env.scene.env_origins
    quat = _quat_wxyz_to_xyzw(_as_torch(robot.data.body_quat_w)[:, idx])
    return pos, quat


def _body_quat(env, body_name: str) -> torch.Tensor:
    _, quat = _body_pose(env, body_name)
    return quat


def _grouped_hand_joint_pos(env) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(
        ABILITY_HAND_TELEOP_JOINT_ORDER, preserve_order=True
    )
    hand_pink = _as_torch(robot.data.joint_pos)[:, joint_ids]
    return hand_pink[:, _TEST_OBS_NEW_GROUPED_FROM_PINK]


def _spine_joint_pos_or_zero(env) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    joint_pos = _as_torch(robot.data.joint_pos)
    values = []
    for joint_name in _TEST_OBS_NEW_SPINE_JOINTS:
        if joint_name in joint_names:
            values.append(joint_pos[:, joint_names.index(joint_name)])
        else:
            values.append(torch.zeros_like(joint_pos[:, 0]))
    return torch.stack(values, dim=1)


def test_obs_new_state(env) -> torch.Tensor:
    """Pack live Alex state into H2Ozone/test_obs_new's 48D state layout."""
    left_pos, left_quat = _body_pose(env, "LEFT_GRIPPER_Z_LINK")
    right_pos, right_quat = _body_pose(env, "RIGHT_GRIPPER_Z_LINK")
    state = torch.cat(
        [
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            _body_quat(env, _TEST_OBS_NEW_LEFT_FOREARM_LINK),
            _body_quat(env, _TEST_OBS_NEW_RIGHT_FOREARM_LINK),
            _body_quat(env, _TEST_OBS_NEW_HEAD_LINK),
            _grouped_hand_joint_pos(env),
            _spine_joint_pos_or_zero(env),
        ],
        dim=1,
    )
    assert state.shape[1] == _TEST_OBS_NEW_STATE_DIM
    return state


def test_obs_new_action(env) -> torch.Tensor:
    """Pack runtime Pink IK actions into H2Ozone/test_obs_new's 46D action layout.

    Arena controls Alex with a 34D tensor: two wrist targets plus 20 hand joints.
    The real-robot dataset action additionally carries forearm and neck quaternion
    groups. Those are not commanded by the Pink IK action term, so we log their live
    same-frame orientations.
    """
    arena_action = env.action_manager.action
    assert arena_action.shape[1] == _TEST_OBS_NEW_ARENA_ACTION_DIM
    wrists = arena_action[:, :14]
    hands_grouped = arena_action[:, 14:34][:, _TEST_OBS_NEW_GROUPED_FROM_PINK]
    action = torch.cat(
        [
            wrists,
            _body_quat(env, _TEST_OBS_NEW_LEFT_FOREARM_LINK),
            _body_quat(env, _TEST_OBS_NEW_RIGHT_FOREARM_LINK),
            _body_quat(env, _TEST_OBS_NEW_HEAD_LINK),
            hands_grouped,
        ],
        dim=1,
    )
    assert action.shape[1] == _TEST_OBS_NEW_ACTION_DIM
    return action


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


class PostStepFlatPolicyActionObservationRecorder(RecorderTerm):
    """Recorder term that records the ``action`` observation group at the end of each step.

    Mirrors the locomanip mimic patch's post-step action recorder, but no-ops on envs
    whose policy does not expose an ``action`` observation group so it can be safely
    enabled for any task.
    """

    def record_post_step(self):
        obs_buf = getattr(self._env, "obs_buf", None)
        if not isinstance(obs_buf, dict) or "action" not in obs_buf:
            return None, None
        return "action", obs_buf["action"]


class PreStepTestObsNewStateRecorder(RecorderTerm):
    """Record Alex state using the H2Ozone/test_obs_new 48D schema."""

    def record_pre_step(self):
        return "observation.state", test_obs_new_state(self._env)


class PreStepTestObsNewActionRecorder(RecorderTerm):
    """Record Alex action using the H2Ozone/test_obs_new 46D schema."""

    def record_pre_step(self):
        return "action", test_obs_new_action(self._env)


@configclass
class PostStepFlatPolicyActionObservationRecorderCfg(RecorderTermCfg):
    """Configuration for the post-step ``action`` observation recorder term."""

    class_type: type[RecorderTerm] = PostStepFlatPolicyActionObservationRecorder


@configclass
class PreStepTestObsNewStateRecorderCfg(RecorderTermCfg):
    """Configuration for the test_obs_new state recorder term."""

    class_type: type[RecorderTerm] = PreStepTestObsNewStateRecorder


@configclass
class PreStepTestObsNewActionRecorderCfg(RecorderTermCfg):
    """Configuration for the test_obs_new action recorder term."""

    class_type: type[RecorderTerm] = PreStepTestObsNewActionRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder manager extended with arena-specific recorder terms."""

    record_pre_step_flat_camera_observations = (
        PreStepFlatCameraObservationsRecorderCfg()
    )
    record_post_step_flat_policy_action_observations = (
        PostStepFlatPolicyActionObservationRecorderCfg()
    )
