# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the PAL Kangaroo."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on running the G1 RL environment."
)
parser.add_argument(
    "--num_envs", type=int, default=10, help="Number of environments to spawn."
)
parser.add_argument(
    "--run_seed", type=int, default=1, help="Number of the run to load."
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

from isaaclab_assets import KANGAROO_MINIMAL_CFG, KANGAROO_FIXED_CFG  # isort: skip
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    """Main function."""
    # create environment configuration
    env_cfg = KangarooFlatEnvCfg()
    env_cfg.scene.robot = KANGAROO_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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

    run_name = f"obs_w_motor_and_measured_joints_seed_{args_cli.run_seed}"
    resume_path = os.path.join(log_root_path, run_name, "model_1999.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # log variables
    lin_track_err = []
    ang_track_err = []

    base_height = []

    proc_actions_list = []
    q_log = []
    qdot_log = []

    # reset environment
    env.reset()
    robot = env.unwrapped.scene.articulations["robot"]

    joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    obs, _ = env.get_observations()

    count = 0
    max_steps = env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)
    # simulate environment
    while simulation_app.is_running() and count < max_steps:
        # run everything in inference mode
        with torch.inference_mode():

            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)

            ref_lin_vel_b = obs[:, 9:11].cpu().detach().numpy()
            ref_ang_vel_b = obs[:, 11:12].cpu().detach().numpy()
            # print("#########################################################")
            # print("ref_lin_vel_b", ref_lin_vel_b.shape)
            # print("robot.data.root_lin_vel_b", robot.data.root_lin_vel_b[:, :2].cpu().detach().numpy().shape)
            # print("ref_ang_vel_b", ref_ang_vel_b.shape)
            # print("robot.data.root_ang_vel_b", robot.data.root_ang_vel_b[:, 2:3].cpu().detach().numpy().shape)

            ang_err = ref_ang_vel_b - robot.data.root_ang_vel_b[:, 2:3].cpu().detach().numpy()
            # print("ang_err", ang_err.shape)
            # print("abs(ang_err)", np.abs(ang_err).shape)

            lin_track_err.append(
                np.linalg.norm(
                    ref_lin_vel_b
                    - robot.data.root_lin_vel_b[:, :2].cpu().detach().numpy(),
                    axis=1,
                )
            )
            ang_track_err.append(
                np.abs(ang_err).squeeze()
            )
            # print("lin_track_err[-1]", lin_track_err[-1].shape)
            # print("ang_track_err[-1]", ang_track_err[-1].shape)

            q_log.append(robot.data.joint_pos.cpu().detach().numpy())
            qdot_log.append(robot.data.joint_vel.cpu().detach().numpy())

            processed_actions = env.unwrapped.action_manager.get_term(
                "joint_pos"
            ).processed_actions
            proc_actions_list.append(processed_actions.cpu().detach().numpy())

            base_height.append(robot.data.root_pos_w[:, 2].cpu().detach().numpy())

            count += 1

    # close the simulator
    env.close()

    lin_track_err = np.array(lin_track_err)
    ang_track_err = np.array(ang_track_err)
    proc_actions_list = np.array(proc_actions_list)
    q_log = np.array(q_log)
    qdot_log = np.array(qdot_log)
    base_height = np.array(base_height)

    # print("lin_track_err", lin_track_err.shape)
    # print("ang_track_err", ang_track_err.shape)
    # print("proc_actions_list", proc_actions_list.shape)
    # print("q_log", q_log.shape)
    # print("qdot_log", qdot_log.shape)
    # print("base_height", base_height.shape)

    np.save(os.path.join(log_root_path, run_name, "lin_track_err.npy"), lin_track_err)
    np.save(os.path.join(log_root_path, run_name, "ang_track_err.npy"), ang_track_err)
    np.save(os.path.join(log_root_path, run_name, "q_ref.npy"), proc_actions_list)
    np.save(os.path.join(log_root_path, run_name, "q.npy"), q_log)
    np.save(os.path.join(log_root_path, run_name, "qdot.npy"), qdot_log)
    np.save(os.path.join(log_root_path, run_name, "base_height.npy"), base_height)


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
