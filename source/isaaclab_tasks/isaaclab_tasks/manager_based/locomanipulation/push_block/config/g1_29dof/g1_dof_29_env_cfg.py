# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


import isaaclab_tasks.manager_based.locomanipulation.push_block.mdp as mdp
from isaaclab_tasks.manager_based.locomanipulation.push_block.push_block_env_cfg import (
    PushBlockCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import G1_29DOF_SPHERE_HANDS_MINIMAL_CFG  # isort: skip


@configclass
class G1Dof29BaseWalkRewards(RewardsCfg):
    """Reward terms for the MDP."""

    # High termination penalty
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Reward tracking linear velocity command
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Reward tracking angular velocity command
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Reward high feet air time for biiped walking
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
            "threshold": 0.7,
        },
    )

    # Penalize base height distance from a given target
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.68,
        },
    )

    # Penalize uneven step times between the two feets
    different_step_times = RewTerm(
        func=mdp.different_step_times,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
        },
    )

    # Penalize too different feet air and contact times
    different_air_contact_times = RewTerm(
        func=mdp.different_air_contact_times,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link"
            ),
        },
    )

    # Penalize small swing feet height
    feet_swing_height = RewTerm(
        func=mdp.feet_swing_height,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "target_height": 0.18,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize sliding feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
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
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_yaw_joint",
                    "waist_pitch_joint",
                    "waist_roll_joint",
                ],
            )
        },
    )


@configclass
class G1Dof29PushBlockRewards(G1Dof29BaseWalkRewards):
    """Reward terms for the MDP."""

    pass


@configclass
class G1Dof29PushBlockEnvCfg(PushBlockCfg):
    rewards: G1Dof29PushBlockRewards = G1Dof29PushBlockRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = G1_29DOF_SPHERE_HANDS_MINIMAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.74),
                joint_pos={
                    ".*_hip_pitch_joint": -0.20,
                    ".*_knee_joint": 0.42,
                    ".*_ankle_pitch_joint": -0.23,
                },
                joint_vel={".*": 0.0},
            ),
        )

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [
            "torso_link"
        ]

        # Rewards
        self.rewards.undesired_contacts.weight = -0.25
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_ankle_roll_link"
        )
        self.rewards.undesired_contacts.params["threshold"] = 400.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-0.1, 0.1)


@configclass
class G1Dof29PushBlockOnlyWalkCfg(G1Dof29PushBlockEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # set the only walk rewards
        self.rewards: G1Dof29BaseWalkRewards = G1Dof29BaseWalkRewards()
        # Rewards
        self.rewards.undesired_contacts.weight = -0.25
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_ankle_roll_link"
        )
        self.rewards.undesired_contacts.params["threshold"] = 400.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # remove block from the scene
        self.scene.block = None

        # track robot velocity
        self.rewards.track_lin_vel_xy_exp.params["asset_cfg"] = SceneEntityCfg("robot")
        self.rewards.track_ang_vel_z_exp.params["asset_cfg"] = SceneEntityCfg("robot")
        self.commands.base_velocity.asset_name = "robot"

        # remove ball-related settings
        self.observations.policy.block_pos = ObsTerm(
            func=mdp.dummy_zero_obs,
            params={"dim": 3},
            noise=Unoise(n_min=-10, n_max=10),
        )
        self.observations.policy.block_quat = ObsTerm(
            func=mdp.dummy_zero_obs,
            params={"dim": 3},
            noise=Unoise(n_min=-10, n_max=10), 
        )
        self.observations.policy.block_vel = ObsTerm(
            func=mdp.dummy_zero_obs,
            params={"dim": 6},
            noise=Unoise(n_min=-10, n_max=10),  # TODO: check this noise size
        )
        self.rewards.block_flat_orientation_l2 = None
        self.events.block_physics_material = None
        self.events.add_block_mass = None
        self.events.reset_block = None

