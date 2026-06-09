# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Upload the latest GR00T checkpoint to a HuggingFace model repo.

With --verify-only, just validates HF_TOKEN and creates the repo (fail-fast
check run before training starts).
"""

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


def find_latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = [
        (int(m.group(1)), p)
        for p in output_dir.iterdir()
        if p.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", p.name))
    ]
    assert checkpoints, f"No checkpoint-* directories found in {output_dir}"
    return max(checkpoints)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--include-optimizer-state",
        action="store_true",
        help="Also upload optimizer/scheduler/rng state (large; only needed to resume training).",
    )
    args = parser.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    print(f"Authenticated to HuggingFace as {user}")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"Model repo ready: {args.repo_id}")

    if args.verify_only:
        return

    assert args.output_dir is not None, "--output-dir is required unless --verify-only"
    checkpoint = find_latest_checkpoint(args.output_dir)
    ignore_patterns = None
    if not args.include_optimizer_state:
        ignore_patterns = ["optimizer.pt", "scheduler.pt", "rng_state*.pth", "global_step*/**"]

    print(f"Uploading {checkpoint} -> {args.repo_id} (path_in_repo={checkpoint.name})")
    api.upload_folder(
        folder_path=str(checkpoint),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=checkpoint.name,
        ignore_patterns=ignore_patterns,
        commit_message=f"Upload {checkpoint.name}",
    )
    print(f"Uploaded: https://huggingface.co/{args.repo_id}/tree/main/{checkpoint.name}")


if __name__ == "__main__":
    main()
