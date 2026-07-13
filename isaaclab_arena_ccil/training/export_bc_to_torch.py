# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the Arena-side metadata + reference pairs for a trained CCIL BC policy.

CCIL's ``train_bc_policy.py`` already saves the greedy policy as a TorchScript module
(``policy.pt``, via d3rlpy ``save_policy``) with the standard/min-max scalers baked in.
That TorchScript file is the artifact Arena consumes directly.

This script produces the companion ``ccil_bc_meta.json`` used to *verify* the artifact
loads and reproduces identical outputs inside Arena's (newer) PyTorch:

* ``input_dim`` / ``output_dim``
* ``ref_pairs``: K sampled observations and the policy's actions on them

It only needs ``torch`` + ``numpy`` (no d3rlpy), so it runs in the CCIL py3.8 env *or* in
the Arena container. If TorchScript fails to load cross-version in Arena, regenerate a
plain ``state_dict`` + full normalization meta from the d3rlpy model (see training/README.md).

Usage (in the CCIL env, after training)::

    python isaaclab_arena_ccil/training/export_bc_to_torch.py \\
        --policy_pt   /path/to/output/policy.pt \\
        --pickle      /datasets/alex_microwave/ccil/alex_microwave.pkl \\
        --out_meta    /datasets/alex_microwave/ccil/ccil_bc_meta.json
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pickle
import sys
import torch

sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _sample_observations(pickle_path: str, num_ref: int, seed: int) -> np.ndarray:
    """Draw num_ref observation rows uniformly across all trajectories."""
    with open(pickle_path, "rb") as f:
        trajectories = pickle.load(f)
    all_obs = np.concatenate([t["observations"] for t in trajectories], axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(all_obs.shape[0], size=min(num_ref, all_obs.shape[0]), replace=False)
    return all_obs[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CCIL BC verification metadata for Arena.")
    parser.add_argument("--policy_pt", required=True, help="TorchScript policy.pt saved by CCIL train_bc_policy.py.")
    parser.add_argument("--pickle", required=True, help="CCIL trajectory pickle (for sampling reference observations).")
    parser.add_argument("--out_meta", required=True, help="Output ccil_bc_meta.json path.")
    parser.add_argument("--num_ref", type=int, default=16, help="Number of reference (obs, action) pairs.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    args = parser.parse_args()

    # d3rlpy save_policy bakes the scaler constants to the training device (typically
    # cuda:0), so the TorchScript module must be loaded/run on CUDA when one is present.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = torch.jit.load(args.policy_pt, map_location=device)
    module.eval()

    obs = _sample_observations(args.pickle, args.num_ref, args.seed)
    with torch.inference_mode():
        actions = module(torch.from_numpy(obs).to(device)).cpu().numpy()

    assert actions.ndim == 2 and actions.shape[0] == obs.shape[0], f"unexpected action shape {actions.shape}"
    assert np.isfinite(actions).all(), "policy produced non-finite actions on reference observations"

    meta = {
        "source": "ccil_save_policy_torchscript",
        "input_dim": int(obs.shape[1]),
        "output_dim": int(actions.shape[1]),
        "num_ref": int(obs.shape[0]),
        "ref_pairs": [
            {"obs": obs[i].tolist(), "action": actions[i].tolist()} for i in range(obs.shape[0])
        ],
    }
    with open(args.out_meta, "w") as f:
        json.dump(meta, f)

    print(f"Wrote {meta['num_ref']} reference pairs to {args.out_meta}")
    print(f"  input_dim={meta['input_dim']}  output_dim={meta['output_dim']}")
    print("  action range:", float(actions.min()), "..", float(actions.max()))


if __name__ == "__main__":
    main()
