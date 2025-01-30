# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from omni.isaac.lab.utils import configclass

from . import custom_shapes
from .shapes_cfg import ShapeCfg


@configclass
class RandomSphereCfg(ShapeCfg):
    """Configuration parameters for a random sphere prim.

    See :meth:`spawn_random_sphere` for more information.
    """

    func: Callable = custom_shapes.spawn_random_sphere

    nominal_radius: float = MISSING
    """Nominal radius of the sphere (in m)."""

    delta_radius: float = MISSING
    """Radius max vcariation around nominal (in m)."""
