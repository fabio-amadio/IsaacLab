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
parser.add_argument(
    "--num_envs", type=int, default=4, help="Number of environments to spawn."
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

import isaaclab_tasks  # noqa: F401


def main():
    """Main function."""
    # create environment configuration
    env_cfg = G1Dof23CatchBallFlatOnlyWalkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    init_state = env_cfg.scene.robot.init_state
    # print(env.scene.articulations["robot"].data.default_joint_pos[0,:])
    # print(init_state)

    # create isaac environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = torch.zeros_like(env.action_manager.action)
            # env stepping
            obs, rew, terminated, truncated, info = env.step(actions)
            print(env.scene.articulations["robot"].body_names)
            # print(env.scene.articulations["robot"].data.root_link_pos_w)
            # print(env.scene.articulations["robot"].data.root_link_pos_w.shape)
            # print(env.scene.rigid_objects["ball"].root_physx_view)
            # print("physics_dt", env.scene.physics_dt)
            
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
