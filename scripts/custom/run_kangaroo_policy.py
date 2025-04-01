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

from isaaclab_tasks.manager_based.locomotion.velocity.config.kangaroo.flat_env_cfg import (
    KangarooFlatEnvCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.kangaroo.agents.rsl_rl_ppo_cfg import (
    KangarooFlatPPORunnerCfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    """Main function."""
    # create environment configuration
    env_cfg = KangarooFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = KangarooFlatPPORunnerCfg()

    # create isaac environment
    env = ManagerBasedRLEnv(env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
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

            robot = env.unwrapped.scene.articulations["robot"]

            raw_actions = env.unwrapped.action_manager.get_term("joint_pos").raw_actions
            processed_actions = env.unwrapped.action_manager.get_term(
                "joint_pos"
            ).processed_actions
            actuated_joint_limits = robot.data.default_joint_pos_limits[
                :, robot.actuators["motor"].joint_indices, :
            ]

            # print("actions", actions.shape)
            # print("raw_actions", raw_actions.shape)
            # print("processed_actions", processed_actions.shape)
            # print("actuated_joint_limits", actuated_joint_limits.shape)

            # Check if raw_actions do not violate actuated_joint_limits
            if not torch.all(
                (raw_actions >= actuated_joint_limits[..., 0])
                & (raw_actions <= actuated_joint_limits[..., 1])
            ):
                print("[WARNING] Raw actions violate actuated joint limits.")

            # Check if processed_actions do not violate actuated_joint_limits
            if not torch.all(
                (processed_actions >= actuated_joint_limits[..., 0])
                & (processed_actions <= actuated_joint_limits[..., 1])
            ):
                print("[WARNING] Processed actions violate actuated joint limits.")



    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
