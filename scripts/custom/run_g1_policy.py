# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the G1."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the G1 RL environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
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

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.locomanipulation.catch_ball.config.g1_23dof.flat_env_cfg import (
    G1Dof23CatchBallFlatEnvCfg,
    G1Dof23CatchBallFlatOnlyWalkEnvCfg,
)

from isaaclab_tasks.manager_based.locomanipulation.catch_ball.config.g1_23dof.agents.rsl_rl_ppo_cfg import (
    G1Dof23CatchBallFlatPPORunnerCfg,
    G1Dof23CatchBallFlatOnlyWalkPPORunnerCfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    """Main function."""
    # create environment configuration
    env_cfg = G1Dof23CatchBallFlatOnlyWalkEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = G1Dof23CatchBallFlatOnlyWalkPPORunnerCfg()

    # create isaac environment
    env = gym.make("Isaac-Catch-Ball-Flat-G1-Dof-23-Only-Walk-v0", cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join(
        "IsaacLab", "logs", "rsl_rl", agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    left_current_air_time = []
    left_last_air_time = []
    right_current_air_time = []
    right_last_air_time = []

    left_current_contact_time = []
    left_last_contact_time = []
    right_current_contact_time = []
    right_last_contact_time = []
    left_step_height = []
    right_step_height = []
    base_height = []

    right_contact_forces = []
    left_contact_forces = [] 
    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            robot = env.unwrapped.scene["robot"]

            # print("physics_dt", env.unwrapped.scene.physics_dt)
            # print("episode_length_buf", env.unwrapped.episode_length_buf)

            # print("left_ankle_roll_link", robot.body_names.index("left_ankle_roll_link"))
            # print("right_ankle_roll_link", robot.body_names.index("right_ankle_roll_link"))


            base_height.append(robot.data.root_pos_w[:, 2].cpu().detach().numpy())

            left_step_height.append(
                robot.data.body_link_pos_w[
                    :, robot.body_names.index("left_ankle_roll_link"), :
                ][:, 2]
                .cpu()
                .detach()
                .numpy()
            )
            right_step_height.append(
                robot.data.body_link_pos_w[
                    :, robot.body_names.index("right_ankle_roll_link"), :
                ][:, 2]
                .cpu()
                .detach()
                .numpy()
            )

            contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
            left_foot_idx = contact_sensor.body_names.index("left_ankle_roll_link")
            right_foot_idx = contact_sensor.body_names.index("right_ankle_roll_link")

            left_current_air_time.append(
                contact_sensor.data.current_air_time[:, left_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )

            right_current_air_time.append(
                contact_sensor.data.current_air_time[:, right_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )

            left_current_contact_time.append(
                contact_sensor.data.current_contact_time[:, left_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )

            right_current_contact_time.append(
                contact_sensor.data.current_contact_time[:, right_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )

            left_last_air_time.append(
                contact_sensor.data.last_air_time[:, left_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )

            right_last_air_time.append(
                contact_sensor.data.last_air_time[:, right_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )

            left_last_contact_time.append(
                contact_sensor.data.last_contact_time[:, left_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )

            right_last_contact_time.append(
                contact_sensor.data.last_contact_time[:, right_foot_idx]
                .cpu()
                .detach()
                .numpy()
            )


            left_contact_forces.append(
                contact_sensor.data.net_forces_w[:, left_foot_idx, :][:, 2]
                .cpu()
                .detach()
                .numpy()
                .squeeze()
            )

            right_contact_forces.append(
                contact_sensor.data.net_forces_w[:, right_foot_idx, :][:, 2]
                .cpu()
                .detach()
                .numpy()
                .squeeze()
            )

    # close the simulator
    env.close()

    base_height = np.array(base_height)
    left_step_height = np.array(left_step_height)
    right_step_height = np.array(right_step_height)
    left_current_air_time = np.array(left_current_air_time)
    left_last_air_time = np.array(left_last_air_time)
    right_current_air_time = np.array(right_current_air_time)
    right_last_air_time = np.array(right_last_air_time)
    left_current_contact_time = np.array(left_current_contact_time)
    left_last_contact_time = np.array(left_last_contact_time)
    right_current_contact_time = np.array(right_current_contact_time)
    right_last_contact_time = np.array(right_last_contact_time)
    left_contact_forces = np.array(left_contact_forces)
    right_contact_forces = np.array(right_contact_forces)


    np.save("base_height.npy", base_height)
    np.save("left_step_height.npy", left_step_height)
    np.save("right_step_height.npy", right_step_height)
    np.save("left_current_air_time.npy", left_current_air_time)
    np.save("left_last_air_time.npy", left_last_air_time)
    np.save("right_current_air_time.npy", right_current_air_time)
    np.save("right_last_air_time.npy", right_last_air_time)
    np.save("left_current_contact_time.npy", left_current_contact_time)
    np.save("left_last_contact_time.npy", left_last_contact_time)
    np.save("right_current_contact_time.npy", right_current_contact_time)
    np.save("right_last_contact_time.npy", right_last_contact_time)
    np.save("left_contact_forces.npy", left_contact_forces)
    np.save("right_contact_forces.npy", right_contact_forces)


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
