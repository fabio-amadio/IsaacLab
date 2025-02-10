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


def block_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the block in the robot frame."""
    block: RigidObject = env.scene[block_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    block_pos_w = block.data.root_link_pos_w
    robot_pos_w = robot.data.root_link_pos_w
    block_to_robot_pos = block_pos_w - robot_pos_w

    return block_to_robot_pos


def ball_rot_in_robot_frame(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The rotation of the block in the robot frame."""
    block: RigidObject = env.scene[block_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    block_quat_w = block.data.root_link_quat_w
    robot_quat_w = robot.data.root_link_quat_w
    block_to_robot_quat = quat_mul(quat_inv(robot_quat_w), block_quat_w)

    return block_to_robot_quat


def ball_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The velocity of the block in the robot frame."""
    block: RigidObject = env.scene[block_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    block_vel_w = block.data.root_link_pos_w
    robot_vel_w = robot.data.root_link_pos_w
    block_to_robot_vel = block_vel_w - robot_vel_w

    return block_to_robot_vel


def dummy_zero_obs(
    env: ManagerBasedRLEnv,
    dim: int,
) -> torch.Tensor:
    """Dummy zero observations."""

    return torch.zeros(env.num_envs, dim, device=env.device)
