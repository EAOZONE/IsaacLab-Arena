# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import torch
import tqdm
from gymnasium.wrappers import RecordVideo
from importlib import import_module
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.camera_video import CameraObsVideoRecorder
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.utils.multiprocess import get_local_rank, get_world_size
from isaaclab_arena.utils.random import set_seed
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.evaluation.perturbation import ArmPoke
    from isaaclab_arena.policy.policy_base import PolicyBase


def get_policy_cls(policy_type: str) -> type["PolicyBase"]:
    """Get the policy class for the given policy type name.

    Note that this function:
    - first: checks for a registered policy type in the PolicyRegistry
    - if not found, it tries to dynamically import the policy class, treating
      the policy_type argument as a string representing the module path and class name.

    """
    from isaaclab_arena.assets.registries import PolicyRegistry

    policy_registry = PolicyRegistry()
    if policy_registry.is_registered(policy_type):
        return policy_registry.get_policy(policy_type)
    else:
        print(f"Policy {policy_type} is not registered. Dynamically importing from path: {policy_type}")
        assert "." in policy_type, (
            "policy_type must be a dotted Python import path of the form 'module.submodule.ClassName', got:"
            f" {policy_type}"
        )
        # Dynamically import the class from the string path
        module_path, class_name = policy_type.rsplit(".", 1)
        module = import_module(module_path)
        policy_cls = getattr(module, class_name)
        return policy_cls


def is_distributed(args_cli: argparse.Namespace) -> bool:
    return (
        "cuda" in args_cli.device and hasattr(args_cli, "distributed") and args_cli.distributed and get_world_size() > 1
    )


def _ik_failed_env_ids(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Subset of ``env_ids`` whose episode ended due to an IK-solver failure.

    Returns an empty tensor unless the env defines the optional ``ik_failure`` termination term
    (enabled by ``--fail_on_ik_error``). The term's done buffer survives ``step()`` (it is filled
    on ``compute`` and not cleared on reset), so it is safe to read here.
    """
    tm = getattr(env.unwrapped, "termination_manager", None)
    if tm is None or "ik_failure" not in tm.active_terms:
        return env_ids[:0]
    mask = tm.get_term("ik_failure")[env_ids].bool()
    return env_ids[mask]


def rollout_policy(
    env,
    policy: "PolicyBase",
    num_steps: int | None,
    num_episodes: int | None,
    language_instruction: str | None = None,
    perturbation: "ArmPoke | None" = None,
    ikstreamer_bridge=None,
    action_target_marker=None,
) -> dict[str, Any]:
    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    pbar = None
    try:
        obs, _ = env.reset()
        policy.reset()
        # Determine language instruction: CLI/job-level override takes precedence over the task's own
        # description. Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing).
        task_description = language_instruction or env.unwrapped.cfg.task_description
        policy.set_task_description(task_description)

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        num_episodes_completed = 0
        num_steps_completed = 0
        num_ik_retries = 0
        episode_step = 0

        # Ramp the poke each episode: episode k (1-based) runs at k x the base wrench.
        # (For a random poke the ramp is disabled; resample() draws a fresh nudge instead.)
        if perturbation is not None:
            perturbation.set_episode_scale(1.0)
            perturbation.resample()

        ikstreamer_dim_mismatch_warned = [False] if ikstreamer_bridge is not None else None

        # Determine if we should stream to IK streamer.
        # If the policy has its own bridge, it handles it internally in get_action.
        # Otherwise, we handle it here in the rollout loop.
        policy_has_bridge = hasattr(policy, "_ikstreamer_bridge")

        while True:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                if action_target_marker is not None:
                    action_target_marker.update(actions)
                if perturbation is not None:
                    perturbation.apply(episode_step)
                obs, _, terminated, truncated, _ = env.step(actions)
                episode_step += 1

                if ikstreamer_bridge is not None and not policy_has_bridge:
                    from isaaclab_arena_gr00t.streaming.gr00t_eef_ikstream_bridge import stream_env_action_to_ikstreamer

                    stream_env_action_to_ikstreamer(
                        ikstreamer_bridge,
                        actions,
                        env=env,
                        dim_mismatch_warned=ikstreamer_dim_mismatch_warned,
                    )

                if terminated.any() or truncated.any():
                    # Only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                    # Per-episode poke counter restarts so each new episode gets poked.
                    episode_step = 0
                    # Episodes that ended due to an IK-solver failure are retried, not scored:
                    # exclude them from the count so we keep going until num_episodes clean episodes.
                    ik_failed_ids = _ik_failed_env_ids(env, env_ids)
                    num_ik_retries += ik_failed_ids.shape[0]
                    # Break if number of episodes is reached
                    completed_episodes = env_ids.shape[0] - ik_failed_ids.shape[0]
                    num_episodes_completed += completed_episodes
                    # Ramp the poke for the upcoming episode (1-based: episode k -> k x base),
                    # or draw a fresh random nudge if the poke is random.
                    if perturbation is not None:
                        perturbation.set_episode_scale(num_episodes_completed + 1)
                        perturbation.resample()
                    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
                        metrics = env.unwrapped.compute_metrics()
                        tqdm.tqdm.write(
                            f"[Rank {get_local_rank()}/{get_world_size()}] Metrics:"
                            f" {metrics_to_plain_python_types(metrics)}"
                        )
                    if num_episodes is not None:
                        pbar.update(completed_episodes)
                        if num_episodes_completed >= num_episodes:
                            break
                # Break if number of steps is reached
                num_steps_completed += 1
                if num_steps is not None:
                    pbar.update(1)
                    if num_steps_completed >= num_steps:
                        break

        pbar.close()
        if num_ik_retries:
            print(f"[Rank {get_local_rank()}/{get_world_size()}] IK-error retries (excluded): {num_ik_retries}")

    except Exception as e:
        if pbar is not None:
            pbar.close()
        raise RuntimeError(f"Error rolling out policy: {e}")

    else:

        # Only compute metrics if env has non-None metrics.
        # Use unwrapped to reach the base env through any gym wrappers (e.g. OrderEnforcing)
        if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
            return env.unwrapped.compute_metrics()
        return None


def main():
    """Run an IsaacLab Arena environment with a policy.
    Use --distributed with torchrun command for one process per GPU on multi-GPU machines. AppLauncher uses LOCAL_RANK for device.
    """
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    local_rank = get_local_rank()
    world_size = get_world_size()
    # Setting device to local rank before SimulationAppContext
    if is_distributed(args_cli):
        args_cli.device = f"cuda:{local_rank}"
        print(f"[Rank {local_rank}/{world_size}] One Isaac Lab instance per process on cuda:{local_rank}")

    with SimulationAppContext(args_cli):

        # Get the policy-type flag before proceeding to other arguments
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()

        # Get the policy class from the policy type
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(
            f"[Rank {local_rank}/{world_size}] Requested policy type: {args_cli.policy_type} -> Policy class:"
            f" {policy_cls}"
        )

        # Add the example environment arguments + policy-related arguments to the parser
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = policy_cls.add_args_to_parser(args_parser)
        args_cli = args_parser.parse_args()
        # Re-apply per-rank device after parse preventing device got overwritten by the default value
        if is_distributed(args_cli):
            args_cli.distributed = True
            args_cli.device = f"cuda:{local_rank}"

        # Build scene. Use rgb_array render mode when recording so RecordVideo can grab frames.
        arena_builder = get_arena_builder_from_cli(args_cli)
        render_mode = "rgb_array" if args_cli.video else None
        env, cfg = arena_builder.make_registered_and_return_cfg(render_mode=render_mode)

        # Optionally shorten episodes so they time out (and reset) sooner. max_episode_length is a
        # live property of episode_length_s, so overriding the cfg here takes effect immediately.
        if args_cli.episode_length_s is not None:
            env.unwrapped.cfg.episode_length_s = args_cli.episode_length_s
            print(
                f"[Rank {local_rank}/{world_size}] Episode length overridden ->"
                f" {args_cli.episode_length_s}s ({env.unwrapped.max_episode_length} steps)"
            )

        # Per-rank seed when distributed so each process has a different seed
        seed = args_cli.seed
        if seed is not None and is_distributed(args_cli):
            seed = seed + local_rank
        if seed is not None:
            set_seed(seed, env)

        # Create the policy from the arguments
        policy = policy_cls.from_args(args_cli)

        # Simulation length.
        if policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"[Rank {local_rank}/{world_size}] Simulation length: {num_episodes} episodes")
            else:
                raise ValueError(f"[Rank {local_rank}/{world_size}] Either num_steps or num_episodes must be provided")

        # Optionally wrap with RecordVideo and/or CameraObsVideoRecorder. The two flags
        # are independent: --video records the kit viewport (via env.render()),
        # --camera_video records the embodiment-mounted cameras (from obs["camera_obs"]).
        if args_cli.video or args_cli.camera_video:
            os.makedirs(args_cli.video_dir, exist_ok=True)
            if num_steps is not None:
                video_length = num_steps
            else:
                # When num_episodes is set, capture exactly one episode's worth of frames.
                # max_episode_length is in environment steps, which matches our rollout cadence.
                video_length = num_episodes * env.unwrapped.max_episode_length

        if args_cli.video:
            env = RecordVideo(
                env,
                video_folder=args_cli.video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
                disable_logger=True,
            )
            print(
                f"[Rank {local_rank}/{world_size}] Recording {video_length}-step viewport video to:"
                f" {args_cli.video_dir}"
            )

        if args_cli.camera_video:
            # Record one mp4 per camera in obs["camera_obs"] (what the policy sees),
            # using the same encoder as RecordVideo.
            env = CameraObsVideoRecorder(
                env,
                video_folder=args_cli.video_dir,
                step_trigger=lambda step: step == 0,
                video_length=video_length,
            )
            print(
                f"[Rank {local_rank}/{world_size}] Recording {video_length}-step per-camera videos to:"
                f" {args_cli.video_dir}"
            )

        # Optionally build an external-force perturbation ("poke") to bump the arm
        # off its expected trajectory during rollout and test policy robustness.
        perturbation = None
        if args_cli.poke:
            from isaaclab_arena.evaluation.perturbation import ArmPoke

            perturbation = ArmPoke(
                env,
                body=args_cli.poke_body,
                force=tuple(args_cli.poke_force),
                torque=tuple(args_cli.poke_torque),
                start_step=args_cli.poke_start_step,
                duration=args_cli.poke_duration,
                period=args_cli.poke_period,
                is_global=(args_cli.poke_frame == "world"),
                show_marker=args_cli.poke_marker,
                random_force=args_cli.poke_random,
                force_range=tuple(args_cli.poke_force_range),
                seed=args_cli.poke_random_seed,
            )
            if args_cli.poke_random:
                force_desc = (
                    f"random horizontal nudge, magnitude {args_cli.poke_force_range[0]}-"
                    f"{args_cli.poke_force_range[1]} N per env/episode"
                )
            else:
                force_desc = f"force={args_cli.poke_force} torque={args_cli.poke_torque} (base wrench; ramped k x on episode k)"
            print(
                f"[Rank {local_rank}/{world_size}] Poke enabled: {force_desc}"
                f" ({args_cli.poke_frame} frame) on {perturbation.body_names}"
                f" for {args_cli.poke_duration} steps starting at step {args_cli.poke_start_step}"
                + (f", repeating every {args_cli.poke_period} steps" if args_cli.poke_period else "")
            )

        # Optionally show spheres at the policy's raw wrist-target positions each step.
        action_target_marker = None
        if args_cli.viz_action_targets:
            from isaaclab_arena.evaluation.action_target_marker import ActionTargetMarker

            action_target_marker = ActionTargetMarker()
            print(f"[Rank {local_rank}/{world_size}] Action-target markers enabled (blue=left, orange=right)")

        steps_str = f"{num_steps} steps" if num_steps is not None else f"{num_episodes} episodes"
        print(f"[Rank {local_rank}/{world_size}] Starting rollout ({steps_str})")

        ikstreamer_bridge = None
        try:
            # NOTE: Gr00tRemoteClosedloopPolicy now handles its own streaming if enabled via CLI args.
            # We only create a bridge here if the policy DOES NOT handle it (e.g. for other policy types).
            if not hasattr(policy, "_ikstreamer_bridge"):
                from isaaclab_arena_gr00t.streaming.gr00t_eef_ikstream_bridge import create_ikstreamer_bridge_from_args

                ikstreamer_bridge = create_ikstreamer_bridge_from_args(args_cli)

            metrics = rollout_policy(
                env,
                policy,
                num_steps,
                num_episodes,
                args_cli.language_instruction,
                perturbation=perturbation,
                ikstreamer_bridge=ikstreamer_bridge,
                action_target_marker=action_target_marker,
            )
        finally:
            if ikstreamer_bridge is not None:
                ikstreamer_bridge.close()

        if metrics is not None:
            print(f"[Rank {local_rank}/{world_size}] Metrics: {metrics_to_plain_python_types(metrics)}")

        # NOTE(huikang, 2025-12-30)Explicitly clean up the remote policy client / server.
        # Do NOT rely on a __del__ destructor in policy for this, since destructors are
        # triggered implicitly and their execution time (or even whether they run)
        # is not guaranteed, which makes resource cleanup unreliable.
        if policy.is_remote:
            policy.shutdown_remote(kill_server=args_cli.remote_kill_on_exit)

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
