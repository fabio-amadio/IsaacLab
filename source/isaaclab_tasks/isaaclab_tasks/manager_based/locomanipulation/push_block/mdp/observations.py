# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_inv

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def block_pos_w(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """The position of the block in world frame."""
    block: RigidObject = env.scene[block_cfg.name]
    block_pos_w = block.data.root_link_pos_w
    return block_pos_w


def block_quat_w(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """The rotation of the block in world frame."""
    block: RigidObject = env.scene[block_cfg.name]
    block_quat_w = block.data.root_link_quat_w
    return block_quat_w


def block_vel_w(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """The velocity of the block in world frame."""
    block: RigidObject = env.scene[block_cfg.name]
    block_vel_w = block.data.root_link_vel_w
    return block_vel_w


def dummy_zero_obs(
    env: ManagerBasedRLEnv,
    dim: int,
) -> torch.Tensor:
    """Dummy zero observations."""

    return torch.zeros(env.num_envs, dim, device=env.device)
