#!/usr/bin/env python3
"""Run the GR00T policy server with deployment-oriented GPU optimizations.

This has the same command-line interface as ``gr00t/eval/run_gr00t_server.py`` but keeps the
deployment knobs outside the Isaac-GR00T submodule. The streaming image uses it for the real
robot path; simulation and upstream GR00T workflows remain unchanged.

Environment variables:
    GR00T_ENABLE_TORCH_COMPILE: Enable ``torch.compile`` for the DiT action head (default: true).
    GR00T_TORCH_COMPILE_MODE: torch.compile mode (default: max-autotune).
    GR00T_DENOISING_STEPS: Optional positive integer overriding the checkpoint's denoising count.
        Lower values reduce latency but alter policy behavior, so the checkpoint default is kept
        unless this variable is deliberately set.
"""

from __future__ import annotations

import json
import os

import torch
import tyro
from gr00t.eval.run_gr00t_server import ServerConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer


def environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false/1/0, got {value!r}")


def configure_policy_for_deployment(policy: Gr00tPolicy) -> None:
    """Apply opt-in quality/latency tuning and eager-to-compiled acceleration."""
    action_head = policy.model.action_head

    denoising_steps = os.environ.get("GR00T_DENOISING_STEPS")
    if denoising_steps is not None:
        requested_steps = int(denoising_steps)
        if requested_steps < 1:
            raise ValueError("GR00T_DENOISING_STEPS must be positive")

        checkpoint_steps = action_head.num_inference_timesteps
        action_head.num_inference_timesteps = requested_steps
        print(
            f"GR00T denoising steps: {checkpoint_steps} -> {requested_steps}. "
            "This reduces latency at the cost of changing the checkpoint's inference behavior."
        )
    else:
        print(f"GR00T denoising steps: checkpoint default ({action_head.num_inference_timesteps})")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if environment_flag("GR00T_ENABLE_TORCH_COMPILE", True):
        compile_mode = os.environ.get("GR00T_TORCH_COMPILE_MODE", "max-autotune")
        print(f"Compiling GR00T DiT action head with torch.compile(mode={compile_mode!r})")
        # The DiT is executed for every denoising iteration. This is the exact target used by
        # Isaac-GR00T's deployment benchmark and avoids compiling the multimodal backbone.
        action_head.model.forward = torch.compile(action_head.model.forward, mode=compile_mode)
    else:
        print("GR00T torch.compile disabled; using eager PyTorch")


def create_policy(config: ServerConfig) -> Gr00tPolicy | ReplayPolicy:
    if config.model_path is not None:
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")

        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        configure_policy_for_deployment(policy)
        return policy

    if config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path) as file:
                modality_configs = json.load(file)
        return ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )

    raise ValueError("Either model_path or dataset_path must be provided")


def main(config: ServerConfig) -> None:
    print("Starting accelerated GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    policy = create_policy(config)
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(policy=policy, host=config.host, port=config.port)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down GR00T inference server...")


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
