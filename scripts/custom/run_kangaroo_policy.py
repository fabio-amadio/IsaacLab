# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the PAL Kangaroo."""

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

from isaaclab_assets import KANGAROO_CFG, KANGAROO_MINIMAL_CFG, KANGAROO_FIXED_CFG  # isort: skip
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    """Main function."""
    # create environment configuration
    env_cfg = KangarooFlatEnvCfg()
    # env_cfg.scene.robot = KANGAROO_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

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

    # log variables

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

    observations_list = []
    raw_actions_list = []
    proc_actions_list = []

    q_log = []
    qdot_log = []

    # reset environment
    env.reset()
    robot = env.unwrapped.scene.articulations["robot"]
    joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    obs, _ = env.get_observations()

    # loaded_actions = np.load("scripts/custom/offline_raw_actions_list.npy")
    # loaded_actions = torch.tensor(loaded_actions, device=env.unwrapped.device)
    # print("loaded_actions", loaded_actions.shape)
    motor_joint_idxs = [4, 11, 18, 19, 36, 37, 38, 40, 43, 45, 46, 48]

    
    actuated_joint_limits = robot.data.default_joint_pos_limits[
        :, robot.actuators["motor"].joint_indices, :
    ]
    default_actuated_joint_pos = robot.data.default_joint_pos[:, motor_joint_idxs]
    print("default motor joint positions")
    print(default_actuated_joint_pos)
    print("actuated joint limits")
    print(actuated_joint_limits)

    count = 0
    max_steps = env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)
    max_steps = 500
    # simulate environment
    while simulation_app.is_running() and count < max_steps:
        # run everything in inference mode
        with torch.inference_mode():

            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # actions = loaded_actions[count, :]

            # log
            observations_list.append(obs.cpu().detach().numpy())

            # env stepping
            obs, _, _, _ = env.step(actions)

            print("root_lin_vel_b", robot.data.root_lin_vel_b)
            print("root_ang_vel_b", robot.data.root_ang_vel_b)

            q_log.append(robot.data.joint_pos.cpu().detach().numpy())
            qdot_log.append(robot.data.joint_vel.cpu().detach().numpy())

            # print("actuators joints names\n", robot.actuators["motor"].joint_names)
            # print("actuators joints indices\n", robot.actuators["motor"].joint_indices)

            raw_actions = env.unwrapped.action_manager.get_term("joint_pos").raw_actions
            processed_actions = env.unwrapped.action_manager.get_term(
                "joint_pos"
            ).processed_actions
            actuated_joint_limits = robot.data.default_joint_pos_limits[
                :, robot.actuators["motor"].joint_indices, :
            ]

            raw_actions_list.append(raw_actions.cpu().detach().numpy())
            proc_actions_list.append(processed_actions.cpu().detach().numpy())

            # print("actions", actions.shape)
            # print("raw_actions", raw_actions.shape)
            # print("processed_actions", processed_actions.shape)
            # print("actuated_joint_limits", actuated_joint_limits.shape)

            # # Check if raw_actions do not violate actuated_joint_limits
            # if not torch.all(
            #     (raw_actions >= actuated_joint_limits[..., 0])
            #     & (raw_actions <= actuated_joint_limits[..., 1])
            # ):
            #     print("[WARNING] Raw actions violate actuated joint limits.")

            # # Check if processed_actions do not violate actuated_joint_limits
            # if not torch.all(
            #     (processed_actions >= actuated_joint_limits[..., 0])
            #     & (processed_actions <= actuated_joint_limits[..., 1])
            # ):
            #     print("[WARNING] Processed actions violate actuated joint limits.")

            # print("raw", raw_actions)
            # print("processed", processed_actions)

            base_height.append(robot.data.root_pos_w[:, 2].cpu().detach().numpy())

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

            contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
            left_foot_idx = contact_sensor.body_names.index("left_ankle_roll")
            right_foot_idx = contact_sensor.body_names.index("right_ankle_roll")
            # print("left_foot_idx", left_foot_idx)
            # print("right_foot_idx", right_foot_idx)

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

            count += 1

    # close the simulator
    env.close()

    observations_list = np.array(observations_list)
    raw_actions_list = np.array(raw_actions_list)
    proc_actions_list = np.array(proc_actions_list)
    q_log = np.array(q_log)
    qdot_log = np.array(qdot_log)
    np.save("scripts/custom/observations_list.npy", observations_list)
    np.save("scripts/custom/raw_actions_list.npy", raw_actions_list)
    np.save("scripts/custom/proc_actions_list.npy", proc_actions_list)
    np.save("scripts/custom/q_log.npy", q_log)
    np.save("scripts/custom/qdot_log.npy", qdot_log)

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

    np.save("scripts/custom/base_height.npy", base_height)
    np.save("scripts/custom/left_step_height.npy", left_step_height)
    np.save("scripts/custom/right_step_height.npy", right_step_height)
    np.save("scripts/custom/left_current_air_time.npy", left_current_air_time)
    np.save("scripts/custom/left_last_air_time.npy", left_last_air_time)
    np.save("scripts/custom/right_current_air_time.npy", right_current_air_time)
    np.save("scripts/custom/right_last_air_time.npy", right_last_air_time)
    np.save("scripts/custom/left_current_contact_time.npy", left_current_contact_time)
    np.save("scripts/custom/left_last_contact_time.npy", left_last_contact_time)
    np.save("scripts/custom/right_current_contact_time.npy", right_current_contact_time)
    np.save("scripts/custom/right_last_contact_time.npy", right_last_contact_time)
    np.save("scripts/custom/left_contact_forces.npy", left_contact_forces)
    np.save("scripts/custom/right_contact_forces.npy", right_contact_forces)


if __name__ == "__main__":
    # run the main function
    main()

    # close sim app
    simulation_app.close()
