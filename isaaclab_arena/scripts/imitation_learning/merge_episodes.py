# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Merge several recorded/generated HDF5 datasets into one (pure h5py, no Isaac Sim).

Concatenates the ``data/demo_*`` episodes from every input file (in the order given) into a
single output file, renumbers them ``demo_0..demo_{N-1}``, recomputes ``data/total``, and
preserves ``env_args``/``format_version`` from the first input. Use it to combine per-door
generated datasets before converting once to LeRobot::

    python isaaclab_arena/scripts/imitation_learning/merge_episodes.py \\
        --output_file /datasets/alex_doorman_all.hdf5 \\
        --input_files /datasets/gen/alex_doorman0_gen.hdf5 \\
                      /datasets/gen/alex_doorman1_gen.hdf5 \\
                      ...
"""

from __future__ import annotations

import argparse

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--input_files", type=str, nargs="+", required=True, help="HDF5 files to merge, in order."
    )
    args = parser.parse_args()

    new_index = 0
    total = 0
    with h5py.File(args.output_file, "w") as dst:
        dst_data = dst.create_group("data")
        attrs_seeded = False

        for input_file in args.input_files:
            with h5py.File(input_file, "r") as src:
                if not attrs_seeded:
                    # Seed root + data-group attrs (env_args, format_version) from the first file.
                    for key, value in src.attrs.items():
                        dst.attrs[key] = value
                    for key, value in src["data"].attrs.items():
                        dst_data.attrs[key] = value
                    attrs_seeded = True

                src_data = src["data"]
                names = list(src_data.keys())
                for name in names:
                    new_name = f"demo_{new_index}"
                    src.copy(src_data[name], dst_data, name=new_name)
                    total += int(dst_data[new_name].attrs.get("num_samples", 0))
                    new_index += 1
                print(f"  {input_file}: +{len(names)} episodes (running total {new_index})")

        dst_data.attrs["total"] = total

    print(f"Merged {new_index} episodes -> {args.output_file}")


if __name__ == "__main__":
    main()
