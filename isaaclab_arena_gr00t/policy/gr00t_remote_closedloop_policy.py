# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote closed-loop policy using GR00T's native PolicyClient.

This policy connects to a GR00T policy server (launched via
``gr00t/eval/run_gr00t_server.py``) and uses its own observation/action translation pipeline.
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import numpy as np
import os
import torch
from dataclasses import dataclass, field
from typing import Any

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.policy.action_scheduling import (
    ActionChunkScheduler,
    ActionScheduler,
    SyncedBatchActionScheduler,
)
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
    Gr00tClosedloopPolicyConfig,
    TaskMode,
)
from isaaclab_arena_gr00t.embodiments.alex.alex_lever_eef_frame import (
    convert_policy_wrist_actions_to_sim,
    convert_sim_eef_state_to_dataset,
    reorder_hand_targets_for_pink_ik,
    uses_lever_eef_frame_bridge,
    write_lever_eef_neck_targets,
)
from isaaclab_arena_gr00t.policy.gr00t_core import (
    Gr00tBasePolicyArgs,
    build_gr00t_action_tensor,
    build_gr00t_policy_observations,
    compute_action_dim,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.utils.io_utils import to_tensor
from isaaclab_arena_gr00t.streaming.gr00t_eef_ikstream_bridge import (
    IKStreamerBridge,
    add_ikstreamer_cli_args,
    create_ikstreamer_bridge_from_args,
    stream_env_action_to_ikstreamer,
)
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    load_gr00t_modality_config_from_file,
)


# TODO(xinjieyao, 2026-04-27): consider adding RemotePolicyArgs to inherit from BasePolicyArgs
# and then having Gr00tRemoteClosedloopPolicyArgs inherit from RemotePolicyArgs
@dataclass
class Gr00tRemoteClosedloopPolicyArgs(Gr00tBasePolicyArgs):
    """Configuration for Gr00tRemoteClosedloopPolicy.

    Inherits policy_config_yaml_path and policy_device from Gr00tBasePolicyArgs,
    and adds remote server connection parameters and num_envs.
    """

    num_envs: int = field(
        default=1, metadata={"help": "Number of environments to simulate"}
    )
    remote_host: str = field(
        default="localhost", metadata={"help": "GR00T policy server hostname"}
    )
    remote_port: int = field(
        default=5555, metadata={"help": "GR00T policy server port"}
    )
    remote_api_token: str | None = field(
        default=None, metadata={"help": "API token for the policy server"}
    )
    stream_ikstreamer: bool = field(
        default=False, metadata={"help": "Mirror actions to RDX IK streamer"}
    )
    ikstreamer_host: str = field(
        default="127.0.0.1", metadata={"help": "RDX IK streamer host"}
    )
    ikstreamer_port: int = field(
        default=2102, metadata={"help": "RDX IK streamer port"}
    )
    debug_ikstreamer: bool = field(
        default=False, metadata={"help": "Print streamed poses to console"}
    )
    ikstreamer_yaw_offset: float = field(
        default=0.0, metadata={"help": "Yaw offset for streamed poses"}
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> Gr00tRemoteClosedloopPolicyArgs:
        """Create configuration from parsed CLI arguments."""
        return cls(
            policy_config_yaml_path=args.policy_config_yaml_path,
            policy_device=args.policy_device,
            num_envs=args.num_envs,
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            remote_api_token=getattr(args, "remote_api_token", None),
            stream_ikstreamer=getattr(args, "stream_ikstreamer", False),
            ikstreamer_host=getattr(args, "ikstreamer_host", "127.0.0.1"),
            ikstreamer_port=getattr(args, "ikstreamer_port", 2102),
            debug_ikstreamer=getattr(args, "debug_ikstreamer", False),
            ikstreamer_yaw_offset=getattr(args, "ikstreamer_yaw_offset", 0.0),
        )


# TODO(xinjieyao, 2026-04-27): add policy registry
class Gr00tRemoteClosedloopPolicy(PolicyBase):
    """GR00T closed-loop policy that delegates inference to a remote GR00T server.

    Uses GR00T's native ``PolicyClient`` (from ``gr00t.policy.server_client``)
    to communicate with a GR00T policy server.
    """

    name = "gr00t_remote_closedloop"
    config_class = Gr00tRemoteClosedloopPolicyArgs

    def __init__(
        self,
        config: Gr00tRemoteClosedloopPolicyArgs,
        action_scheduler_cls: type[ActionScheduler] = ActionChunkScheduler,
    ):
        super().__init__(config)

        # Policy config (for obs/action translation — no model loading)
        # TODO(xinjieyao, 2026-04-27): to be refactored
        self.policy_config: Gr00tClosedloopPolicyConfig = create_config_from_yaml(
            config.policy_config_yaml_path, Gr00tClosedloopPolicyConfig
        )
        self.num_envs = config.num_envs
        self.device = config.policy_device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        # Joint configs (for sim from/to policy joint space remapping)
        (
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.robot_state_joints_config,
        ) = load_gr00t_joint_configs(self.policy_config)

        self.modality_configs = load_gr00t_modality_config_from_file(
            self.policy_config.modality_config_path,
            self.policy_config.embodiment_tag,
        )

        # Action / chunk shapes
        self.action_dim = compute_action_dim(
            self.task_mode, self.robot_action_joints_config
        )
        self.action_chunk_length = self.policy_config.action_chunk_length

        self._chunking_state: ActionScheduler | None = action_scheduler_cls(
            num_envs=self.num_envs,
            action_chunk_length=self.action_chunk_length,
            action_horizon=self.policy_config.action_horizon,
            action_dim=self.action_dim,
            device=self.device,
            dtype=torch.float,
        )

        # Connect to GR00T's native PolicyClient
        client = Gr00tPolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        self._client: Gr00tPolicyClient | None = client
        if not client.ping():
            raise ConnectionError(
                f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}"
            )

        self.task_description: str | None = None

        # Optional RDX IK streamer
        self._ikstreamer_bridge: IKStreamerBridge | None = (
            create_ikstreamer_bridge_from_args(config)
        )
        self._ikstreamer_dim_mismatch_warned = [False]

        self._use_lever_eef_frame_bridge = uses_lever_eef_frame_bridge(
            self.policy_config.modality_config_path
        )
        self._neck_action_chunk: torch.Tensor | None = None

        # Policy-rate action pacing: the chunk is trained at ``policy_control_hz`` (e.g. 30) but the
        # sim steps faster (e.g. 50 Hz). We set the scheduler's zero-order-hold rate on the first
        # step (when the env's ``step_dt`` is known) so the chunk plays at real speed.
        self._policy_control_hz = getattr(self.policy_config, "policy_control_hz", None)
        self._action_rate_set = False

        # ``robot_joint_pos`` is emitted in Isaac Sim's articulation joint order, which is NOT
        # the block order of ``state_joints_config_path`` (legs/arms/hands interleave left/right
        # and hand q1s precede q2s). Reading state columns by the static YAML index therefore
        # scrambles the hand/neck state sent to GR00T. We rebuild the name->column map from the
        # live ``robot.joint_names`` on the first step and use it for all state indexing.
        self._live_state_joints_config: dict[str, int] | None = None

        # Opt-in diagnostics: set GR00T_DEBUG_STATE=1 to dump, on the first few chunks,
        # the exact state groups sent to the server and the live sim joint order. This is
        # for verifying that Arena's observation matches the training dataset convention
        # (joint order / frame / units). Disabled by default; no behavior change.
        self._debug_state = os.environ.get("GR00T_DEBUG_STATE", "") not in ("", "0")
        self._debug_state_calls_left = int(os.environ.get("GR00T_DEBUG_STATE_CALLS", "2"))
        self._debug_joint_order_dumped = False
        # GR00T_DEBUG_ACTIONS=1 logs the per-step applied wrist targets and marks chunk
        # boundaries, to diagnose non-smooth (retracting) motion across chunk stitches.
        self._debug_actions = os.environ.get("GR00T_DEBUG_ACTIONS", "") not in ("", "0")
        self._debug_action_step = 0

    # ---------------------- CLI helpers -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            "Gr00t Remote Closedloop Policy",
            "Arguments for GR00T remote closed-loop policy evaluation.",
        )
        group.add_argument(
            "--policy_config_yaml_path",
            type=str,
            required=True,
            help="Path to the Gr00t closedloop policy config YAML file",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device for Arena-side tensor operations (default: cuda)",
        )
        group.add_argument(
            "--remote_host",
            type=str,
            default="localhost",
            help="GR00T policy server hostname",
        )
        group.add_argument(
            "--remote_port", type=int, default=5555, help="GR00T policy server port"
        )
        group.add_argument(
            "--remote_api_token",
            type=str,
            default=None,
            help="API token for the policy server",
        )
        add_ikstreamer_cli_args(parser)
        group.add_argument(
            "--scheduler",
            type=str,
            default="chunk",
            choices=["chunk", "synced_batch"],
            help=(
                "Action scheduler: 'chunk' fetches a new chunk for any env that needs one;"
                " 'synced_batch' waits until ALL envs need a new chunk and then issues a single"
                " full-batch inference call (envs that finish early hold their current robot state)."
            ),
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> Gr00tRemoteClosedloopPolicy:
        config = Gr00tRemoteClosedloopPolicyArgs.from_cli_args(args)
        scheduler_cls: type[ActionScheduler] = (
            SyncedBatchActionScheduler
            if getattr(args, "scheduler", "chunk") == "synced_batch"
            else ActionChunkScheduler
        )
        return Gr00tRemoteClosedloopPolicy(config, action_scheduler_cls=scheduler_cls)

    # ---------------------- Policy interface -------------------

    def set_task_description(self, task_description: str | None) -> str:
        if task_description is None:
            task_description = self.policy_config.language_instruction
        if not task_description:
            raise ValueError(
                "No language instruction provided. Set 'language_instruction' in the job config, "
                "pass --language_instruction on the CLI, or define 'task_description' on the task class."
            )
        self.task_description = task_description
        return self.task_description

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        assert self._chunking_state is not None, "GR00T remote policy has been closed"

        self._resolve_state_joints_config(env)
        self._maybe_set_action_rate(env)

        def fetch_chunk() -> torch.Tensor:
            return self._get_action_chunk(
                observation,
                self.policy_config.pov_cam_name_sim,
                env=env,
            )

        actions = self._chunking_state.get_action(
            fetch_chunk,
            hold_action=self._extract_hold_action(observation),
        )

        if self._use_lever_eef_frame_bridge and self._neck_action_chunk is not None:
            self._write_neck_for_current_step(env)

        if self._ikstreamer_bridge is not None:
            stream_env_action_to_ikstreamer(
                self._ikstreamer_bridge,
                actions,
                env=env,
                env_index=0,
                dim_mismatch_warned=self._ikstreamer_dim_mismatch_warned,
            )

        if self._debug_actions:
            idx = int(self._chunking_state.current_action_index[0].item())
            boundary = " <== NEW CHUNK" if idx in (0, 1) else ""
            a = actions[0].detach().cpu().numpy()
            print(
                f"[GR00T_DEBUG_ACTIONS] step={self._debug_action_step:3d} chunk_idx={idx:2d}"
                f" Lwrist_pos={np.round(a[0:3],4)} Rwrist_pos={np.round(a[7:10],4)}{boundary}",
                flush=True,
            )
            self._debug_action_step += 1

        return actions

    def _active_state_joints_config(self) -> dict[str, int]:
        """State joint name->column map to index ``robot_joint_pos`` with.

        Prefers the live map built from ``robot.joint_names`` (correct order); falls back to
        the static YAML only before the first ``get_action`` has resolved the env.
        """
        return self._live_state_joints_config or self.robot_state_joints_config

    def _maybe_set_action_rate(self, env: gym.Env | None) -> None:
        """Pace chunk consumption to ``policy_control_hz`` given the sim's step rate.

        Called once (env is needed for ``step_dt``). No-op when ``policy_control_hz`` is unset or
        the scheduler doesn't support rate control (e.g. synced-batch).
        """
        if self._action_rate_set or self._policy_control_hz is None or env is None:
            return
        self._action_rate_set = True
        scheduler = self._chunking_state
        set_rate = getattr(scheduler, "set_action_rate", None)
        if set_rate is None:
            print(
                "[Gr00tRemoteClosedloopPolicy] policy_control_hz set but scheduler does not support"
                " rate pacing; ignoring."
            )
            return
        step_dt = float(getattr(env, "unwrapped", env).step_dt)
        sim_hz = 1.0 / step_dt
        sim_steps_per_action = sim_hz / float(self._policy_control_hz)
        set_rate(sim_steps_per_action)
        print(
            f"[Gr00tRemoteClosedloopPolicy] action pacing: sim={sim_hz:.1f}Hz,"
            f" policy={self._policy_control_hz:.1f}Hz -> {sim_steps_per_action:.3f} sim steps/waypoint"
        )

    def _resolve_state_joints_config(self, env: gym.Env | None) -> None:
        """Build and cache the state joint name->column map from the live articulation order.

        ``robot_joint_pos`` columns follow ``robot.joint_names`` (Isaac Sim articulation order),
        not the static ``state_joints_config`` YAML order. We rebuild the map once so that state
        remapping and hold-action indexing read each joint from its true column.
        """
        if self._live_state_joints_config is not None or env is None:
            return
        robot = getattr(env, "unwrapped", env).scene["robot"]
        sim_names = list(robot.joint_names)
        live = {name: i for i, name in enumerate(sim_names)}
        missing = [n for n in self.robot_state_joints_config if n not in live]
        assert not missing, (
            f"state_joints_config joints missing from sim articulation: {missing}"
        )
        self._live_state_joints_config = live

    def _extract_hold_action(self, observation: dict[str, Any]) -> torch.Tensor:
        """Build the action vector that waiting envs should hold: their current sim joint positions
        copied into the action slots that share a joint name with the state config."""
        joint_pos_sim = observation["policy"]["robot_joint_pos"].to(
            device=self.device, dtype=torch.float
        )
        hold_action = torch.zeros(
            (self.num_envs, self.action_dim), dtype=torch.float, device=self.device
        )
        state_joints_config = self._active_state_joints_config()
        for joint_name, action_idx in self.robot_action_joints_config.items():
            state_idx = state_joints_config.get(joint_name)
            if state_idx is not None:
                hold_action[:, action_idx] = joint_pos_sim[:, state_idx]
        return hold_action

    def _consumed_chunk_steps(self) -> torch.Tensor:
        """Per-env index of the action step just returned by the action scheduler."""
        assert self._chunking_state is not None, "GR00T remote policy has been closed"
        idx = self._chunking_state.current_action_index.clone()
        step = idx - 1
        step[idx == -1] = self.action_chunk_length - 1
        return step.clamp(min=0)

    def _write_neck_for_current_step(self, env: gym.Env) -> None:
        assert self._neck_action_chunk is not None
        steps = self._consumed_chunk_steps()
        batch_idx = torch.arange(self.num_envs, device=self.device)
        neck_targets = self._neck_action_chunk[batch_idx, steps]
        env_mask = getattr(self._chunking_state, "env_requires_new_chunk", None)
        write_lever_eef_neck_targets(env, neck_targets, env_mask=env_mask)

    def _get_action_chunk(
        self,
        observation: dict[str, Any],
        camera_names: list[str] | str = "robot_head_cam_rgb",
        env: gym.Env | None = None,
    ) -> torch.Tensor:
        """Get an action chunk from the remote GR00T server.

        Calls GR00T's PolicyClient to get the action chunk.
        """
        if isinstance(camera_names, str):
            camera_names = [camera_names]

        # 1. Reuse the same obs translation as local policy
        assert self.task_description is not None, "Task description is not set"
        assert self._client is not None, "GR00T remote policy has been closed"
        rgb_list_np, joint_pos_sim_np, eef_pose_np = extract_obs_numpy_from_torch(
            nested_obs=observation, camera_names=camera_names
        )
        if self._use_lever_eef_frame_bridge and env is not None and eef_pose_np:
            eef_pose_np = convert_sim_eef_state_to_dataset(eef_pose_np, env)
        policy_observations = build_gr00t_policy_observations(
            rgb_list_np=rgb_list_np,
            joint_pos_sim_np=joint_pos_sim_np,
            task_description=self.task_description,
            policy_config=self.policy_config,
            robot_state_joints_config=self._active_state_joints_config(),
            policy_joints_config=self.policy_joints_config,
            modality_configs=self.modality_configs,
            eef_pose_policy=eef_pose_np,
        )

        if self._debug_state and self._debug_state_calls_left > 0:
            self._dump_debug_state(policy_observations, env)
            self._debug_state_calls_left -= 1

        # 2. Call GR00T's own client
        robot_action_policy, _ = self._client.get_action(policy_observations)

        if self._use_lever_eef_frame_bridge and env is not None:
            robot_action_policy = convert_policy_wrist_actions_to_sim(
                robot_action_policy, env
            )

        if "neck" in robot_action_policy:
            self._neck_action_chunk = to_tensor(
                robot_action_policy["neck"], device=self.device
            )
        else:
            self._neck_action_chunk = None

        # 3. Action translation from policy output to sim action tensor
        action_tensor = build_gr00t_action_tensor(
            robot_action_policy=robot_action_policy,
            task_mode=self.task_mode,
            policy_joints_config=self.policy_joints_config,
            robot_action_joints_config=self.robot_action_joints_config,
            device=self.device,
            embodiment_tag=self.policy_config.embodiment_tag,
        )

        if self._use_lever_eef_frame_bridge and env is not None:
            # Semantic (real-robot) hand targets must be permuted into the order the
            # Pink IK action term actually applies its hand block in.
            action_tensor = reorder_hand_targets_for_pink_ik(action_tensor, env)

        assert (
            action_tensor.shape[0] == self.num_envs
            and action_tensor.shape[1] >= self.action_chunk_length
        )
        return action_tensor

    # lever_eef dataset (H2Ozone/lever_eef) per-group STATE means, for OOD comparison.
    _DEBUG_LEVER_EEF_STATE_MEANS = {
        "left_wrist_pose": [-0.3476, 0.4902, 1.0489, 0.1074, -0.6494, 0.1085, 0.7427],
        "right_wrist_pose": [-0.1924, -0.1967, 1.1776, 0.1255, -0.7544, 0.2047, 0.5935],
        "left_hand": [0.604, 1.362, 0.599, 1.357, 0.601, 1.360, 0.388, 1.134, -0.508, 0.004],
        "right_hand": [1.199, 1.993, 1.204, 1.998, 1.204, 1.998, 1.204, 1.998, -0.236, 0.042],
        "neck": [-0.2208, 0.4832],
    }

    def _dump_debug_state(self, policy_observations: dict[str, Any], env: gym.Env | None) -> None:
        """Print the state groups sent to the server (env 0) vs dataset means, plus sim joint order.

        Enabled with ``GR00T_DEBUG_STATE=1``. Purely diagnostic; helps confirm whether the
        state Arena feeds matches the training dataset convention (joint order / frame / units).
        """
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        _p = lambda *a: print(*a, flush=True)  # noqa: E731 - flush so output isn't stuck in pipe buffer
        _p("\n========== [GR00T_DEBUG_STATE] state sent to server (env 0) ==========")
        state = policy_observations.get("state", {})
        for key, arr in state.items():
            values = np.asarray(arr)[0, 0]
            ref = self._DEBUG_LEVER_EEF_STATE_MEANS.get(key)
            line = f"[{key}] sent = {values}"
            if ref is not None and len(ref) == len(values):
                ref_arr = np.asarray(ref)
                line += f"\n    {'dataset_mean':>12} = {ref_arr}"
                line += f"\n    {'abs_diff':>12} = {np.abs(values - ref_arr)}  (max {np.abs(values - ref_arr).max():.4f})"
            _p(line)

        if not self._debug_joint_order_dumped and env is not None:
            self._debug_joint_order_dumped = True
            try:
                robot = getattr(env, "unwrapped", env).scene["robot"]
                sim_names = list(robot.joint_names)
                cfg_names = sorted(
                    self.robot_state_joints_config, key=self.robot_state_joints_config.get
                )
                _p("\n========== [GR00T_DEBUG_STATE] sim joint order vs state config ==========")
                _p(f"sim robot.joint_names ({len(sim_names)}):\n{sim_names}")
                mismatches = [
                    (i, cfg_names[i], sim_names[i])
                    for i in range(min(len(sim_names), len(cfg_names)))
                    if cfg_names[i] != sim_names[i]
                ]
                if len(sim_names) != len(cfg_names):
                    _p(
                        f"LENGTH MISMATCH: sim has {len(sim_names)} joints, "
                        f"state config has {len(cfg_names)}."
                    )
                if mismatches:
                    _p(
                        f"ORDER MISMATCH at {len(mismatches)} positions "
                        f"(idx: config_name != sim_name) -> state is SCRAMBLED:"
                    )
                    for i, cfg_n, sim_n in mismatches:
                        _p(f"    [{i:2d}] config={cfg_n} != sim={sim_n}")
                else:
                    _p("Joint order matches state config positionally (no scramble). OK")
            except Exception as e:  # noqa: BLE001 - diagnostics must never break rollout
                _p(f"[GR00T_DEBUG_STATE] could not read sim joint order: {e}")
        _p("=====================================================================\n")

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        assert self._client is not None, "GR00T remote policy has been closed"
        assert self._chunking_state is not None, "GR00T remote policy has been closed"
        self._client.reset()
        self._chunking_state.reset(env_ids)
        self._neck_action_chunk = None

    def close(self) -> None:
        """Release Arena-side resources for the remote GR00T policy client."""
        if self._ikstreamer_bridge is not None:
            self._ikstreamer_bridge.close()
            self._ikstreamer_bridge = None

        client = self._client
        try:
            if client is not None:
                socket = getattr(client, "socket", None)
                context = getattr(client, "context", None)
                try:
                    if socket is not None:
                        socket.close(linger=0)
                finally:
                    if context is not None:
                        context.term()
        finally:
            self._client = None
            self._chunking_state = None
            self.modality_configs = None
