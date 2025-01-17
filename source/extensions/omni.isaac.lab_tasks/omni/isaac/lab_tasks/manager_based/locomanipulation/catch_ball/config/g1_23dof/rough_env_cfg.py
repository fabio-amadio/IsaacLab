# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.locomanipulation.catch_ball.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomanipulation.catch_ball.catch_ball_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import G1_23DOF_RUBBER_HANDS_MINIMAL_CFG  # isort: skip


@configclass
class G1dof23CatchBallRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "threshold": 0.4, 
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            )
        },
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_roll_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )


@configclass
class G1dof23CatchBallRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1dof23CatchBallRewards = G1dof23CatchBallRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = G1_23DOF_RUBBER_HANDS_MINIMAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.74),
                joint_pos={
                    ".*_hip_pitch_joint": -0.20,
                    ".*_knee_joint": 0.42,
                    ".*_ankle_pitch_joint": -0.23,
                    "left_shoulder_roll_joint": 0.12,
                    "right_shoulder_roll_joint": -0.12,
                    "left_shoulder_yaw_joint": -0.32,
                    "right_shoulder_yaw_joint": 0.32,
                    "left_wrist_roll_joint": -1.3,
                    "right_wrist_roll_joint": 1.3,
                },
                joint_vel={".*": 0.0},
            ),
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            "torso_link"
        ]
        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link", ".*_hand"]

        # # ------ DEACTIVATE BALL-RELATED TERMS ------
        # self.scene.ball = None
        # self.observations.policy.ball_pos = ObsTerm(
        #     func=mdp.dummy_zero_obs, params={"dim": 3}
        # )
        # self.observations.policy.ball_vel = ObsTerm(
        #     func=mdp.dummy_zero_obs, params={"dim": 3}
        # )
        # self.rewards.ball_close_to_body = None
        # self.rewards.ball_close_to_hands = None
        # self.rewards.dropping_ball = None
        # self.events.reset_ball = None
        # self.terminations.ball_dropped


@configclass
class G1dof23CatchBallRoughEnvCfg_PLAY(G1dof23CatchBallRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
