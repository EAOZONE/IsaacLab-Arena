# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Psyonic official → IHMC Ability Hand URDF compatibility."""

import urllib.request
import xml.etree.ElementTree as ET

import pytest

from isaaclab_arena.embodiments.alex.ability_hand_urdf import (
    AbilityHandPackageLayout,
    detect_package_layout,
    materialize_ihmc_compatible_hand_urdf,
    resolve_models_dir_candidates,
)
from isaaclab_arena.embodiments.alex.alex import ABILITY_HAND_JOINT_NAMES_LIST


def _official_left_urdf_url() -> str:
    return (
        "https://raw.githubusercontent.com/psyonicinc/ability-hand-ros2/main/"
        "src/ah_urdf/urdf/ability_hand_left_large.urdf"
    )


@pytest.fixture(scope="module")
def official_psyonic_hand_tree(tmp_path_factory) -> ET.ElementTree:
    cache = tmp_path_factory.mktemp("psyonic_hand")
    urdf_path = cache / "ability_hand_left_large.urdf"
    if not urdf_path.exists():
        urllib.request.urlretrieve(_official_left_urdf_url(), urdf_path)
    models_dir = cache
    (models_dir / "models").mkdir(exist_ok=True)
    # materialize requires models/ to exist even though meshes are not loaded in these tests.
    output_path = cache / "ability_hand_left_large_ihmc_compat.urdf"
    materialize_ihmc_compatible_hand_urdf(
        str(urdf_path),
        "left",
        str(output_path),
        models_dir=str(models_dir),
    )
    return ET.parse(output_path)


def test_detect_psyonic_layout(tmp_path):
    root = tmp_path / "ah_urdf"
    (root / "urdf").mkdir(parents=True)
    (root / "urdf" / "ability_hand_left_large.urdf").write_text(
        '<robot name="ability_hand"><joint name="index_q1"/></robot>'
    )
    assert detect_package_layout(str(root)) == AbilityHandPackageLayout.PSYONIC


def test_detect_ihmc_layout(tmp_path):
    root = tmp_path / "ihmc_hands_ros2"
    (root / "urdf" / "abilityHand").mkdir(parents=True)
    (root / "urdf" / "abilityHand" / "ability_hand_left_large.urdf").write_text(
        '<robot name="left_ability_hand"><joint name="left_ability_hand_index_q1"/></robot>'
    )
    assert detect_package_layout(str(root)) == AbilityHandPackageLayout.IHMC


def test_resolve_models_dir_prefers_psyonic(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    alex_models = sdk / "alex-models"
    alex_models.mkdir(parents=True)
    (alex_models / "alex_V1_description").mkdir()

    psyonic = sdk / "ability-hand-ros2" / "src" / "ah_urdf" / "urdf"
    psyonic.mkdir(parents=True)
    (psyonic / "ability_hand_left_large.urdf").write_text(
        '<robot name="ability_hand"><joint name="index_q1"/></robot>'
    )

    ihmc = sdk / "alex-ros2" / "ihmc_hands_ros2" / "urdf" / "abilityHand"
    ihmc.mkdir(parents=True)
    (ihmc / "ability_hand_left_large.urdf").write_text(
        '<robot name="left_ability_hand"><joint name="left_ability_hand_index_q1"/></robot>'
    )

    monkeypatch.delenv("ABILITY_HAND_MODELS_DIR", raising=False)
    from isaaclab_arena.embodiments.alex import ability_hand_urdf

    resolved = ability_hand_urdf.resolve_models_dir(str(alex_models))
    assert resolved.endswith("ah_urdf")
    assert detect_package_layout(resolved) == AbilityHandPackageLayout.PSYONIC


def test_materialized_joint_names_match_arena_list(official_psyonic_hand_tree):
    root = official_psyonic_hand_tree.getroot()
    urdf_joint_names = {joint.get("name") for joint in root.findall("joint")}
    left_joints = {name for name in ABILITY_HAND_JOINT_NAMES_LIST if name.startswith("left_")}
    assert left_joints.issubset(urdf_joint_names)


def test_materialized_base_link_matches_adapter(official_psyonic_hand_tree):
    root = official_psyonic_hand_tree.getroot()
    link_names = {link.get("name") for link in root.findall("link")}
    assert "left_ability_hand_base" in link_names
    assert "base_link" not in link_names


def test_materialized_finger_q2_mimic_uses_ihmc_coefficients(official_psyonic_hand_tree):
    root = official_psyonic_hand_tree.getroot()
    q2 = root.find(".//joint[@name='left_ability_hand_index_q2']/mimic")
    assert q2 is not None
    assert float(q2.get("multiplier")) == pytest.approx(1.05851325)
    assert float(q2.get("offset")) == pytest.approx(0.72349796)
    assert q2.get("joint") == "left_ability_hand_index_q1"


def test_materialized_finger_q1_limit_matches_ihmc(official_psyonic_hand_tree):
    root = official_psyonic_hand_tree.getroot()
    limit = root.find(".//joint[@name='left_ability_hand_index_q1']/limit")
    assert limit is not None
    assert float(limit.get("upper")) == pytest.approx(1.74)


def test_resolve_models_dir_candidates_includes_psyonic_paths():
    paths = resolve_models_dir_candidates("/models/alex-models")
    joined = "\n".join(paths)
    assert "ability-hand-ros2" in joined
    assert "ihmc_hands_ros2" in joined
