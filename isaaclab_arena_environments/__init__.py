# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Example environments for Isaac Lab-Arena.

Environment modules are registered lazily via :func:`isaaclab_arena_environments.cli.ensure_environments_registered`
so importing this package (or ``cli``) does not pull every environment module — and anything that
transitively imports ``pxr`` — before ``SimulationApp`` starts.
"""

from __future__ import annotations

import importlib
import pkgutil

_NON_ENVIRONMENT_MODULES = {"cli", "example_environment_base", "lever_scene_builder"}

_environments_registered = False


def ensure_environments_registered() -> None:
    """Import every ``@register_environment`` module once to populate the registry."""
    global _environments_registered
    if _environments_registered:
        return
    for _importer, _modname, _ispkg in pkgutil.iter_modules(__path__):
        if not _ispkg and _modname not in _NON_ENVIRONMENT_MODULES:
            importlib.import_module(f"{__name__}.{_modname}")
    _environments_registered = True
