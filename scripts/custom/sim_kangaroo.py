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


KANGAROO_FIXED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://localhost/Users/jlcucumber/Kangaroo_!/kangaroo/kangaroo_customed_2.usd", # f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/PAL/Kangaroo/kangaroo.usd"
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "leg_left_.*": 0.0,
            "leg_right_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "motor": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_left_1_motor",
                "leg_right_1_motor",
                "leg_left_2_motor",
                "leg_left_3_motor",
                "leg_right_2_motor",
                "leg_right_3_motor",
                "leg_left_4_motor",
                "leg_left_5_motor",
                "leg_left_length_motor",
                "leg_right_length_motor",
                "leg_right_4_motor",
                "leg_right_5_motor",
            ],
            effort_limit={
                "leg_left_1_motor": 3000,  # 2000,
                "leg_right_1_motor": 3000,  # 2000,
                "leg_left_2_motor": 3000,
                "leg_right_2_motor": 3000,
                "leg_left_3_motor": 3000,
                "leg_right_3_motor": 3000,
                "leg_left_4_motor": 3000,
                "leg_right_4_motor": 3000,
                "leg_left_5_motor": 3000,
                "leg_right_5_motor": 3000,
                "leg_left_length_motor": 5000,
                "leg_right_length_motor": 5000,
            },
            velocity_limit={
                "leg_left_1_motor": 0.4,
                "leg_right_1_motor": 0.4,
                "leg_left_2_motor": 0.4,
                "leg_right_2_motor": 0.4,
                "leg_left_3_motor": 0.4,
                "leg_right_3_motor": 0.4,
                "leg_left_4_motor": 0.4,
                "leg_right_4_motor": 0.4,
                "leg_left_5_motor": 0.4,
                "leg_right_5_motor": 0.4,
                "leg_left_length_motor": 0.625,
                "leg_right_length_motor": 0.625,
            },
            stiffness={
                "leg_left_1_motor":         200000.0,
                "leg_right_1_motor":        200000.0,
                "leg_left_2_motor":         200000.0,
                "leg_right_2_motor":        200000.0,
                "leg_left_3_motor":         200000.0,
                "leg_right_3_motor":        200000.0,
                "leg_left_4_motor":         500000.0,
                "leg_right_4_motor":        500000.0,
                "leg_left_5_motor":         500000.0,
                "leg_right_5_motor":        500000.0,
                "leg_left_length_motor":    500000.0,
                "leg_right_length_motor":   500000.0,
            },
            damping={
                "leg_left_1_motor":         1500.0,
                "leg_right_1_motor":        1500.0,
                "leg_left_2_motor":         1500.0,
                "leg_right_2_motor":        1500.0,
                "leg_left_3_motor":         1500.0,
                "leg_right_3_motor":        1500.0,
                "leg_left_4_motor":         2500.0,
                "leg_right_4_motor":        2500.0,
                "leg_left_5_motor":         2500.0,
                "leg_right_5_motor":        2500.0,
                "leg_left_length_motor":    2500.0,
                "leg_right_length_motor":   2500.0,
            },
            armature={
                "leg_left_1_motor": 0.01,
                "leg_right_1_motor": 0.01,
                "leg_left_2_motor": 0.01,
                "leg_right_2_motor": 0.01,
                "leg_left_3_motor": 0.01,
                "leg_right_3_motor": 0.01,
                "leg_left_4_motor": 0.01,
                "leg_right_4_motor": 0.01,
                "leg_left_5_motor": 0.01,
                "leg_right_5_motor": 0.01,
                "leg_left_length_motor": 0.01,
                "leg_right_length_motor": 0.01,
            },
        ),
    },
)


def main():
    """Main function."""
    # create environment configuration
    env_cfg = KangarooFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 5.0
    env_cfg.scene.robot = KANGAROO_FIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_motor"],
        scale=1.0,
        use_default_offset=False,
    )

    init_state = env_cfg.scene.robot.init_state
    # print(env.scene.articulations["robot"].data.default_joint_pos[0,:])

    # create isaac environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene.articulations["robot"]
    actuated_joint_limits = robot.data.default_joint_pos_limits[
        :, robot.actuators["motor"].joint_indices, :
    ]
    print("actuators joints names\n", robot.actuators["motor"].joint_names)
    print("actuators joints indices\n", robot.actuators["motor"].joint_indices)
    print("actuated_joint_limits\n", actuated_joint_limits)
    env.reset()

    # dump the list of robot.joint_names into a CSV file
    with open("joint_names.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([[name] for name in robot.joint_names])
    # print(f"CSV Path : {os.path.abspath('joint_names.csv')}")

    # initialize action variables
    actions = torch.zeros_like(env.action_manager.action)
    for i, q in enumerate(
        env.scene.articulations["robot"].data.joint_pos[
            0, robot.actuators["motor"].joint_indices
        ]
    ):
        actions[0, i] = q
    print("actions", actions)
    a_idx = 8  # pick between 0 and 11
    a_step = 0.001
    a_step_dir = 1

    # init log lists
    a_log = []
    q_log = []
    qdot_log = []

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

            # env stepping
            obs, rew, terminated, truncated, info = env.step(actions)

            count += 1

    # close the simulator
    env.close()

    print()
    a_log = np.array(a_log)
    q_log = np.array(q_log)
    qdot_log = np.array(qdot_log)
    # save the logs
    np.save("a_log.npy", a_log)
    np.save("q_log.npy", q_log)
    np.save("qdot_log.npy", qdot_log)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
