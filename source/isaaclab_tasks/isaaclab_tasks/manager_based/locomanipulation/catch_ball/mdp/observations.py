# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, DeformableObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_inv

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def ball_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the ball in the robot frame."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    ball_pos_w = ball.data.root_link_pos_w
    robot_pos_w = robot.data.root_link_pos_w
    ball_to_robot_pos = ball_pos_w - robot_pos_w

    return ball_to_robot_pos


def soft_ball_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the soft ball in the robot frame."""
    ball: DeformableObject = env.scene[ball_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    ball_pos_w = ball.data.root_pos_w
    robot_pos_w = robot.data.root_link_pos_w
    ball_to_robot_pos = ball_pos_w - robot_pos_w

    return ball_to_robot_pos


def ball_rot_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The rotation of the ball in the robot frame."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    ball_quat_w = ball.data.root_link_quat_w
    robot_quat_w = robot.data.root_link_quat_w
    ball_to_robot_quat = quat_mul(quat_inv(robot_quat_w), ball_quat_w)

    return ball_to_robot_quat


def ball_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The velocity of the ball in the robot frame."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    ball_vel_w = ball.data.root_link_vel_w
    robot_vel_w = robot.data.root_link_vel_w
    ball_to_robot_vel = ball_vel_w - robot_vel_w

    return ball_to_robot_vel


def soft_ball_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The velocity of the soft ball in the robot frame."""
    ball: DeformableObject = env.scene[ball_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]

    ball_vel_w = ball.data.root_vel_w
    robot_vel_w = robot.data.root_link_vel_w
    ball_to_robot_vel = ball_vel_w - robot_vel_w

    return ball_to_robot_vel


def dummy_zero_obs(
    env: ManagerBasedRLEnv,
    dim: int,
) -> torch.Tensor:
    """Dummy zero observations."""

    return torch.zeros(env.num_envs, dim, device=env.device)
