# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drop episodes from a recorded/annotated HDF5 dataset by index (pure h5py, no Isaac Sim).

Episode indices match the order printed by ``replay_demos.py`` (``data`` group key order),
so review with ``replay_demos.py`` first, note the bad indices, then::

    python isaaclab_arena/scripts/imitation_learning/filter_episodes.py \\
        --input_file  /datasets/alex_doorman0.hdf5 \\
        --output_file /datasets/alex_doorman0_clean.hdf5 \\
        --remove 3 7 12

Survivors are renumbered ``demo_0..demo_{N-1}`` and the ``data/total`` step count is recomputed.
Use ``--keep`` instead of ``--remove`` to keep only the listed indices.
"""

from __future__ import annotations

import argparse

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--remove", type=int, nargs="+", help="Episode indices to drop.")
    group.add_argument("--keep", type=int, nargs="+", help="Episode indices to keep (drop the rest).")
    args = parser.parse_args()

    with h5py.File(args.input_file, "r") as src, h5py.File(args.output_file, "w") as dst:
        # Preserve root attrs (e.g. format_version).
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        src_data = src["data"]
        dst_data = dst.create_group("data")
        # Preserve data-group attrs (env_args, ...); recompute total below.
        for key, value in src_data.attrs.items():
            dst_data.attrs[key] = value

        # Same ordering replay_demos.py indexes by (h5py key order).
        names = list(src_data.keys())
        if args.keep is not None:
            keep_indices = set(args.keep)
        else:
            keep_indices = set(range(len(names))) - set(args.remove)

        total = 0
        new_index = 0
        for index, name in enumerate(names):
            if index not in keep_indices:
                print(f"  dropping #{index} ({name})")
                continue
            new_name = f"demo_{new_index}"
            src.copy(src_data[name], dst_data, name=new_name)
            total += int(dst_data[new_name].attrs.get("num_samples", 0))
            new_index += 1

        dst_data.attrs["total"] = total

    print(f"Wrote {new_index} episodes (dropped {len(names) - new_index}) -> {args.output_file}")


if __name__ == "__main__":
    main()
