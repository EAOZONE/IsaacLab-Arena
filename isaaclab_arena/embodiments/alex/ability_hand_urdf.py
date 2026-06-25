# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve Psyonic Ability Hand URDFs and convert official naming to the IHMC/Arena convention.

Official Psyonic ``ability-hand-ros2`` URDFs use short link/joint names (``index_q1``,
``base``) and ``package://ah_urdf/models/`` STL meshes.  Arena's Alex stack (adapters,
Pink IK, dex-retargeting, GR00T joint order) expects the IHMC-prefixed names from
``ihmc_hands_ros2`` (``left_ability_hand_index_q1``, etc.).  This module materialises
IHMC-compatible hand URDFs from either package layout.
"""

from __future__ import annotations

import os
import re
from enum import Enum

# 4-bar finger q2 mimic — matches ihmc_hands_ros2 and ability_hand_joint_expand_retargeter.
_ABILITY_HAND_Q2_MIMIC_MULTIPLIER = 1.05851325
_ABILITY_HAND_Q2_MIMIC_OFFSET = 0.72349796

# Joint limits used throughout Arena (IHMC convention); official Psyonic URDFs differ.
_FINGER_Q1_LIMIT = (0.0, 1.74)
_FINGER_Q2_LIMIT = (0.766, 2.61)
_THUMB_Q1_LIMIT = (-1.74, 0.0)
_THUMB_Q2_LIMIT = (0.0, 1.74)

_PSYONIC_URDF_DIR = os.path.join("urdf")
_IHMC_URDF_DIR = os.path.join("urdf", "abilityHand")


class AbilityHandPackageLayout(str, Enum):
    """Which Ability Hand model package layout ``models_dir`` points at."""

    PSYONIC = "psyonic"
    IHMC = "ihmc"


def hand_urdf_filename(side: str, *, large: bool = True) -> str:
    assert side in ("left", "right"), f"side must be 'left' or 'right', got {side!r}"
    size = "large" if large else "small"
    return f"ability_hand_{side}_{size}.urdf"


def detect_package_layout(models_dir: str) -> AbilityHandPackageLayout | None:
    """Return the detected layout, or ``None`` if ``models_dir`` has no hand URDFs."""
    for side in ("left", "right"):
        if _psyonic_source_path(models_dir, side) is not None:
            return AbilityHandPackageLayout.PSYONIC
        if _ihmc_source_path(models_dir, side) is not None:
            return AbilityHandPackageLayout.IHMC
    return None


def resolve_models_dir_candidates(alex_models_dir: str) -> list[str]:
    """Search paths for Ability Hand packages (Psyonic preferred, then IHMC)."""
    if explicit := os.environ.get("ABILITY_HAND_MODELS_DIR"):
        return [os.path.normpath(explicit)]

    sdk_root = os.path.dirname(alex_models_dir)
    return [
        os.path.join(sdk_root, "ability-hand-ros2", "src", "ah_urdf"),
        os.path.join(sdk_root, "ah_urdf"),
        "/ability-hand-ros2/src/ah_urdf",
        os.path.join(sdk_root, "alex-ros2", "ihmc_hands_ros2"),
        "/ihmc_hands_ros2",
    ]


def resolve_models_dir(alex_models_dir: str) -> str:
    """Pick the first valid Ability Hand package root on the candidate search path."""
    for candidate in resolve_models_dir_candidates(alex_models_dir):
        root = os.path.normpath(candidate)
        if detect_package_layout(root) is not None:
            return root
    return os.path.normpath(resolve_models_dir_candidates(alex_models_dir)[0])


def resolve_source_hand_urdf(models_dir: str, side: str) -> tuple[str, AbilityHandPackageLayout]:
    """Return ``(absolute_urdf_path, layout)`` for one hand."""
    psyonic = _psyonic_source_path(models_dir, side)
    if psyonic is not None:
        return psyonic, AbilityHandPackageLayout.PSYONIC
    ihmc = _ihmc_source_path(models_dir, side)
    assert ihmc is not None, (
        f"Ability Hand URDF for side={side!r} not found under {models_dir}.\n"
        "Mount official Psyonic ability-hand-ros2 (recommended):\n"
        "  export ABILITY_HAND_MODELS_DIR=/path/to/ability-hand-ros2/src/ah_urdf\n"
        "Or mount ihmc-alex-sdk (legacy ihmc_hands_ros2):\n"
        "  ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk"
    )
    return ihmc, AbilityHandPackageLayout.IHMC


def materialize_ihmc_compatible_hand_urdf(
    source_path: str,
    side: str,
    output_path: str,
    *,
    models_dir: str,
) -> str:
    """Write an IHMC-named hand URDF, converting from Psyonic official layout when needed."""
    marker = (
        f"<!-- ability_hand_compat source={source_path} side={side} "
        f"models_dir={models_dir} layout={AbilityHandPackageLayout.PSYONIC.value} -->"
    )
    if (
        os.path.exists(output_path)
        and os.path.getmtime(output_path) >= os.path.getmtime(source_path)
        and marker in open(output_path, encoding="utf-8").read(512)
    ):
        return output_path

    from lxml import etree

    tree = etree.parse(source_path)
    root = tree.getroot()
    name_map = _build_ihmc_name_map(side)
    mesh_prefix = _mesh_models_dir(models_dir)

    root.set("name", f"{side}_ability_hand")

    for element in list(root):
        tag = element.tag
        old_name = element.get("name", "")
        if old_name in name_map and name_map[old_name] is None:
            root.remove(element)
            continue
        if tag == "link" and old_name in name_map:
            new_name = name_map[old_name]
            assert new_name is not None
            element.set("name", new_name)
        elif tag == "joint":
            if old_name in name_map:
                new_name = name_map[old_name]
                if new_name is None:
                    root.remove(element)
                    continue
                element.set("name", new_name)
            for ref_tag in ("parent", "child"):
                ref = element.find(ref_tag)
                if ref is not None:
                    link_name = ref.get("link", "")
                    if link_name in name_map:
                        mapped = name_map[link_name]
                        if mapped is not None:
                            ref.set("link", mapped)
            mimic = element.find("mimic")
            if mimic is not None:
                mimic_joint = mimic.get("joint", "")
                if mimic_joint in name_map and name_map[mimic_joint] is not None:
                    mimic.set("joint", name_map[mimic_joint])
            _apply_ihmc_joint_limits(element, side)

        for mesh in element.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn.startswith("package://ah_urdf/models/"):
                mesh.set("filename", mesh_prefix + fn[len("package://ah_urdf/models/") :])
            elif fn.startswith("package://abilityHand/"):
                mesh.set(
                    "filename",
                    os.path.join(models_dir, "meshes", "abilityHand") + "/" + fn[len("package://abilityHand/") :],
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n{marker}\n")
    return output_path


def ensure_arena_hand_urdf(
    models_dir: str,
    side: str,
    output_dir: str,
    *,
    suffix: str = "ihmc_compat",
) -> str:
    """Return an IHMC-compatible hand URDF path (materialising from Psyonic when needed)."""
    source_path, layout = resolve_source_hand_urdf(models_dir, side)
    if layout == AbilityHandPackageLayout.IHMC:
        return source_path

    output_path = os.path.join(output_dir, f"ability_hand_{side}_large_{suffix}.urdf")
    return materialize_ihmc_compatible_hand_urdf(
        source_path,
        side,
        output_path,
        models_dir=models_dir,
    )


def _psyonic_source_path(models_dir: str, side: str) -> str | None:
    path = os.path.join(models_dir, _PSYONIC_URDF_DIR, hand_urdf_filename(side))
    if os.path.isfile(path) and not _urdf_uses_ihmc_names(path):
        return path
    return None


def _ihmc_source_path(models_dir: str, side: str) -> str | None:
    path = os.path.join(models_dir, _IHMC_URDF_DIR, hand_urdf_filename(side))
    if os.path.isfile(path) and _urdf_uses_ihmc_names(path):
        return path
    return None


def _urdf_uses_ihmc_names(urdf_path: str) -> bool:
    with open(urdf_path, encoding="utf-8") as handle:
        head = handle.read(4096)
    return f"{_side_from_urdf_path(urdf_path)}_ability_hand_index_q1" in head


def _side_from_urdf_path(urdf_path: str) -> str:
    match = re.search(r"ability_hand_(left|right)_", os.path.basename(urdf_path))
    assert match is not None, f"Cannot infer hand side from {urdf_path!r}"
    return match.group(1)


def _mesh_models_dir(models_dir: str) -> str:
    models = os.path.join(models_dir, "models")
    assert os.path.isdir(models), (
        f"Psyonic Ability Hand mesh directory not found: {models}\n"
        "Expected ability-hand-ros2 layout: src/ah_urdf/models/"
    )
    return models + "/"


def _build_ihmc_name_map(side: str) -> dict[str, str | None]:
    prefix = f"{side}_ability_hand_"
    mapping: dict[str, str | None] = {
        "base": f"{prefix}base",
        "base_link": None,
        "base_link_to_base": None,
    }
    for finger in ("index", "middle", "ring", "pinky", "thumb"):
        mapping[f"{finger}_L1"] = f"{prefix}{finger}_L1"
        mapping[f"{finger}_L2"] = f"{prefix}{finger}_L2"
        mapping[f"{finger}_q1"] = f"{prefix}{finger}_q1"
        mapping[f"{finger}_q2"] = f"{prefix}{finger}_q2"
        mapping[f"{finger}_anchor"] = f"{prefix}{finger}_anchor"
    mapping.update(
        {
            "idx_anchor": f"{prefix}idx_anchor",
            "mid_anchor": f"{prefix}mid_anchor",
            "rng_anchor": f"{prefix}rng_anchor",
            "pnky_anchor": f"{prefix}pnky_anchor",
            "thmb_anchor": f"{prefix}thmb_anchor",
        }
    )
    for index in range(30):
        mapping[f"fsr{index}"] = f"{prefix}fsr{index}"
    return mapping


def _apply_ihmc_joint_limits(joint_element, side: str) -> None:
    from lxml import etree

    name = joint_element.get("name", "")
    prefix = f"{side}_ability_hand_"
    if not name.startswith(prefix):
        return

    limit = joint_element.find("limit")
    if limit is None:
        return

    suffix = name[len(prefix) :]
    if suffix in ("index_q1", "middle_q1", "ring_q1", "pinky_q1"):
        lower, upper = _FINGER_Q1_LIMIT
    elif suffix in ("index_q2", "middle_q2", "ring_q2", "pinky_q2"):
        lower, upper = _FINGER_Q2_LIMIT
        mimic = joint_element.find("mimic")
        if mimic is None:
            mimic = etree.SubElement(joint_element, "mimic")
        mimic.set("joint", f"{prefix}{suffix.replace('_q2', '_q1')}")
        mimic.set("multiplier", str(_ABILITY_HAND_Q2_MIMIC_MULTIPLIER))
        mimic.set("offset", str(_ABILITY_HAND_Q2_MIMIC_OFFSET))
    elif suffix == "thumb_q1":
        lower, upper = _THUMB_Q1_LIMIT
    elif suffix == "thumb_q2":
        lower, upper = _THUMB_Q2_LIMIT
    else:
        return

    limit.set("lower", str(lower))
    limit.set("upper", str(upper))
