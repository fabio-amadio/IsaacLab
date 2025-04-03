# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the G1."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

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

from isaaclab_tasks.manager_based.locomanipulation.catch_ball.config.g1_23dof.flat_env_cfg import (
    G1Dof23CatchBallFlatEnvCfg,
    G1Dof23CatchBallFlatOnlyWalkEnvCfg,
)


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


G1_23DOF_FIXED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/G1/g1_23dof_rubber_hands_minimal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.84),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_yaw_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_elbow_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)


def main():
    """Main function."""
    # create environment configuration
    env_cfg = G1Dof23CatchBallFlatOnlyWalkEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 5.0
    env_cfg.scene.robot = G1_23DOF_FIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # create isaac environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    # robot = env.scene.articulations["robot"]
    env.reset()

    # initialize action variables
    actions = torch.zeros_like(env.action_manager.action)
    a_idx = 11  # pick between 0 and 11
    a_step = 0.01
    a_step_dir = 1

    count = 0
    max_steps = env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)
    # simulate environment
    while simulation_app.is_running() and count < max_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # env stepping
            obs, rew, terminated, truncated, info = env.step(actions)
            actions[0, a_idx] = actions[
                :, a_idx
            ] + a_step_dir * a_step * torch.ones_like(actions[0, a_idx])
            # print(env.scene.articulations["robot"].body_names)
            # print(env.scene.articulations["robot"].data.root_link_pos_w)
            # print(env.scene.articulations["robot"].data.root_link_pos_w.shape)
            # print(env.scene.rigid_objects["ball"].root_physx_view)
            # print("physics_dt", env.scene.physics_dt)
            count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
