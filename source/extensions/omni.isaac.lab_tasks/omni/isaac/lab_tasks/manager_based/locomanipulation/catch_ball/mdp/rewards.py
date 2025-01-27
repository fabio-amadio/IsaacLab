# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, DeformableObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import (
    quat_rotate_inverse,
    yaw_quat,
    matrix_from_quat,
    quat_error_magnitude,
)

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_air_time_positive_biped(
    env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


def feet_slide(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(
        yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def same_hands_orientation_exp(
    env,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for keeping its hands in the same orientation using exponential kernel."""
    robot = env.scene[robot_cfg.name]

    left_hand_rot = robot.data.body_link_quat_w[
        :, robot.body_names.index("left_wrist_roll_rubber_hand"), :
    ]
    right_hand_rot = robot.data.body_link_quat_w[
        :, robot.body_names.index("right_wrist_roll_rubber_hand"), :
    ]

    # Compute the difference in orientation between the hands using quaternion error magnitude
    orientation_diff = quat_error_magnitude(left_hand_rot, right_hand_rot)

    return torch.exp(-orientation_diff / std**2)


def ball_close_to_body_exp(
    env,
    std: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding the ball close to the body using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    ball: RigidObject = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    ball_pos = ball.data.root_link_pos_w
    robot_pos = robot.data.root_link_pos_w
    ball_to_robot_pos = ball_pos - robot_pos
    return torch.exp(-torch.square(ball_to_robot_pos).sum(dim=1) / std**2)


def ball_close_to_hands_exp(
    env,
    std: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding the ball in the robot's hand using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    ball: RigidObject = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    ball_pos = ball.data.root_link_pos_w
    left_hand_pos = robot.data.body_link_pos_w[
        :, robot.body_names.index("left_wrist_roll_rubber_hand"), :
    ]
    left_hand_rot = robot.data.body_link_quat_w[
        :, robot.body_names.index("left_wrist_roll_rubber_hand"), :
    ]
    left_hand_rot_mat = matrix_from_quat(left_hand_rot)
    left_palm_pos = left_hand_pos + left_hand_rot_mat @ torch.tensor(
        [0.17, 0.0, 0.0], device=env.device
    )

    right_hand_pos = robot.data.body_link_pos_w[
        :, robot.body_names.index("right_wrist_roll_rubber_hand"), :
    ]
    right_hand_rot = robot.data.body_link_quat_w[
        :, robot.body_names.index("right_wrist_roll_rubber_hand"), :
    ]
    right_hand_rot_mat = matrix_from_quat(right_hand_rot)
    right_palm_pos = right_hand_pos + right_hand_rot_mat @ torch.tensor(
        [0.17, 0.0, 0.0], device=env.device
    )

    ball_to_left_pos = ball_pos - left_palm_pos
    ball_to_right_pos = ball_pos - right_palm_pos
    return torch.exp(
        (
            -torch.square(ball_to_left_pos).sum(dim=1)
            - torch.square(ball_to_right_pos).sum(dim=1)
        )
        / std**2
    )


def soft_ball_close_to_hands_exp(
    env,
    std: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding the (soft) ball in the robot's hand using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    ball: DeformableObject = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    ball_pos = ball.data.root_pos_w
    left_hand_pos = robot.data.body_link_pos_w[
        :, robot.body_names.index("left_wrist_roll_rubber_hand"), :
    ]
    left_hand_rot = robot.data.body_link_quat_w[
        :, robot.body_names.index("left_wrist_roll_rubber_hand"), :
    ]
    left_hand_rot_mat = matrix_from_quat(left_hand_rot)
    left_palm_pos = left_hand_pos + left_hand_rot_mat @ torch.tensor(
        [0.17, 0.0, 0.0], device=env.device
    )

    right_hand_pos = robot.data.body_link_pos_w[
        :, robot.body_names.index("right_wrist_roll_rubber_hand"), :
    ]
    right_hand_rot = robot.data.body_link_quat_w[
        :, robot.body_names.index("right_wrist_roll_rubber_hand"), :
    ]
    right_hand_rot_mat = matrix_from_quat(right_hand_rot)
    right_palm_pos = right_hand_pos + right_hand_rot_mat @ torch.tensor(
        [0.17, 0.0, 0.0], device=env.device
    )

    ball_to_left_pos = ball_pos - left_palm_pos
    ball_to_right_pos = ball_pos - right_palm_pos
    return torch.exp(
        (
            -torch.square(ball_to_left_pos).sum(dim=1)
            - torch.square(ball_to_right_pos).sum(dim=1)
        )
        / std**2
    )


def ball_speed(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize ball velocity when grasped (SE kernel used to filter with ball-robot distance)."""
    # extract the used quantities (to enable type-hinting)
    ball = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_link_lin_vel_w
    return ball_vel.norm(dim=1)


def dropping_ball(
    env,
    threshold: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Penalize dropping the ball."""
    # extract the used quantities (to enable type-hinting)
    ball = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_link_pos_w
    return (ball_pos[:, 2] < threshold).float()


def angular_momentum_exp(
    env,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward small angular momentum."""
    # extract the used quantities (to enable type-hinting)
    robot = env.scene[robot_cfg.name]
    # Default masses. Shape is (num_instances, num_bodies).
    body_masses = robot.data.default_mass.to(env.device)
    # Default inertias. Shape is (num_instances, num_bodies, 9).
    body_inertias = robot.data.default_inertia.to(env.device)
    # State of all bodies CoM [pos, quat, lin_vel, ang_vel] in world frame.
    # Shape is (num_instances, num_bodies, 13).
    body_com_states = robot.data.body_com_state_w.to(env.device)

    # Compute the total angular momentum
    angular_momentum = torch.zeros(
        (robot.data.body_com_state_w.shape[0], 3), device=env.device
    )
    for i in range(robot.data.body_com_state_w.shape[1]):
        mass = body_masses[:, i].unsqueeze(-1)
        inertia = body_inertias[:, i].view(-1, 3, 3)
        com_pos = body_com_states[:, i, :3]
        com_ang_vel = body_com_states[:, i, 10:13]
        com_lin_vel = body_com_states[:, i, 7:10]

        # Compute linear momentum
        lin_momentum = mass * com_lin_vel

        # Compute angular momentum
        ang_momentum = torch.cross(com_pos, lin_momentum) + torch.bmm(
            inertia, com_ang_vel.unsqueeze(-1)
        ).squeeze(-1)

        # Sum the angular momentum of all bodies
        angular_momentum += ang_momentum

    # print("## mean angular_momentum: ", torch.mean(torch.norm(angular_momentum, dim=1)))
    # print("## mean reward: ", torch.mean(torch.exp(-torch.norm(angular_momentum, dim=1) / std**2)))

    return torch.exp(-torch.norm(angular_momentum, dim=1) / std**2)
