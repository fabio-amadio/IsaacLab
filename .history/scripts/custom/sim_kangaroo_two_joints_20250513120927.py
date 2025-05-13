# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the Kangaroo."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
import csv

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the G1 RL environment."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import pdb

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.kangaroo.flat_env_cfg import (
    KangarooFlatEnvCfg_PLAY,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab_assets import (
    KANGAROO_CFG,
    KANGAROO_MINIMAL_CFG,
    KANGAROO_FIXED_CFG,
)  # isort: skip


def main():
    """Main function."""
    # create environment configuration
    env_cfg = KangarooFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1.0
    env_cfg.scene.robot = KANGAROO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_motor"],
        scale=1.0,
        use_default_offset=False,
    )

    # init_state = env_cfg.scene.robot.init_state

    # create isaac environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene.articulations["robot"]
    actuated_joint_limits = robot.data.default_joint_pos_limits[
        :, robot.actuators["motor"].joint_indices, :
    ]
    # print("actuators joints names\n", robot.actuators["motor"].joint_names)
    # print("actuators joints indices\n", robot.actuators["motor"].joint_indices)
    # print("actuated_joint_limits\n", actuated_joint_limits)

    print(
        "default_joint_pos",
        robot.data.default_joint_pos[:, robot.actuators["motor"].joint_indices],
    )
    print("joint_pos", robot.data.joint_pos[:, robot.actuators["motor"].joint_indices])
    env.reset()
    # # reset dof state
    # print("##########################################################################")
    # print(len(robot.data.joint_names))
    # print(robot.data.joint_names)
    # print("##########################################################################")
    joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # dump the list of robot.joint_names into a CSV file
    with open("joint_names.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([[name] for name in robot.joint_names])

    # initialize action variables
    actions = torch.zeros_like(env.action_manager.action)
    for i, q in enumerate(
        env.scene.articulations["robot"].data.default_joint_pos[
            0, robot.actuators["motor"].joint_indices
        ]
    ):
        actions[0, i] = q
    print("actions", actions)
    a_idx = 8  # pick between 0 and 11
    a_step = 0.0
    a_step_dir = 1

    # init log lists
    a_log = []
    q_log = []
    qdot_log = []
    left_contact_forces = []
    right_contact_forces = []
    left_step_height = []
    right_step_height = []

    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    left_foot_idx = contact_sensor.body_names.index("left_ankle_roll")
    right_foot_idx = contact_sensor.body_names.index("right_ankle_roll")

    count = 0
    max_steps = env_cfg.episode_length_s * 1000 / (env_cfg.sim.dt * env_cfg.decimation)
    # simulate environment
    while simulation_app.is_running() and count < max_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # env.sim.step()
            # env.scene.update(dt=env_cfg.sim.dt)

            # if end of range is reached, reverse the increment direction
            if a_step_dir == 1 and actions[
                0, a_idx
            ] + a_step_dir * a_step * torch.ones_like(
                actions[0, a_idx]
            ) >= actuated_joint_limits[
                0, a_idx, 1
            ] * torch.ones_like(
                actions[0, a_idx]
            ):
                a_step_dir = -1

            if a_step_dir == -1 and actions[
                0, a_idx
            ] + a_step_dir * a_step * torch.ones_like(
                actions[0, a_idx]
            ) <= actuated_joint_limits[
                0, a_idx, 0
            ] * torch.ones_like(
                actions[0, a_idx]
            ):
                a_step_dir = 1

            # compute next action (incremental)
            actions[0, a_idx] = actions[
                :, a_idx
            ] + a_step_dir * a_step * torch.ones_like(actions[0, a_idx])

            robot_unwr = env.unwrapped.scene["robot"]
            a_log.append(actions.cpu().detach().numpy())
            q_log.append(robot_unwr.data.joint_pos.cpu().detach().numpy())
            qdot_log.append(robot_unwr.data.joint_vel.cpu().detach().numpy())

            # print("actions", actions.shape)
            # print("joint_names", robot.joint_names)
            # print("joint_pos", robot.data.joint_pos)
            # print("body_names", robot.data.body_names)
            # print("actuators\n", robot.actuators["motor"])
            # print(
            #     "default_joint_pos_limits\n",
            #     robot.data.default_joint_pos_limits,
            # )

            left_contact_forces.append(
                contact_sensor.data.net_forces_w[:, left_foot_idx, :][:, :]
                .cpu()
                .detach()
                .numpy()
                .squeeze()
            )

            right_contact_forces.append(
                contact_sensor.data.net_forces_w[:, right_foot_idx, :][:, :]
                .cpu()
                .detach()
                .numpy()
                .squeeze()
            )

            left_step_height.append(
                robot.data.body_link_pos_w[
                    :, robot.body_names.index("left_ankle_roll"), :
                ][:, 2]
                .cpu()
                .detach()
                .numpy()
            )
            right_step_height.append(
                robot.data.body_link_pos_w[
                    :, robot.body_names.index("right_ankle_roll"), :
                ][:, 2]
                .cpu()
                .detach()
                .numpy()
            )


            # env stepping
            obs, rew, terminated, truncated, info = env.step(actions)

            count += 1

    # close the simulator
    env.close()

    print()
    a_log = np.array(a_log)
    q_log = np.array(q_log)
    qdot_log = np.array(qdot_log)
    left_contact_forces = np.array(left_contact_forces)
    right_contact_forces = np.array(right_contact_forces)
    left_step_height = np.array(left_step_height)
    right_step_height = np.array(right_step_height)
    # save the logs
    np.save("a_log.npy", a_log)
    np.save("q_log.npy", q_log)
    np.save("qdot_log.npy", qdot_log)
    np.save("left_contact_forces.npy", left_contact_forces)
    np.save("left_step_height.npy", left_step_height)
    np.save("right_step_height.npy", right_step_height)

    # Create a dictionary with joint names as keys and their positions as values
    joint_positions = {
        name: pos.item()
        for name, pos in zip(robot.joint_names, robot.data.joint_pos[0])
    }
    print("Joint Positions Dictionary:\n", joint_positions)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
