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


import isaaclab_tasks.manager_based.locomanipulation.catch_ball.mdp as mdp
from isaaclab_tasks.manager_based.locomanipulation.catch_ball.catch_ball_env_cfg import (
    CatchBallAndLocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import G1_23DOF_RUBBER_HANDS_MINIMAL_CFG  # isort: skip


@configclass
class G1BaseWalkRewards(RewardsCfg):
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
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Reward high feet air time for biiped walking
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

    # Penalize base height distance from a given target
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.25,
        params={
            "target_height": 0.73,
        },
    )

    # Penalize uneven step times between the two feets
    different_step_times = RewTerm(
        func=mdp.different_step_times,
        weight=-0.25,
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
        weight=-0.25,
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
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "target_height": 0.15,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_ankle_roll_link", preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize sliding feet
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
class G1CatchBallRewards(G1BaseWalkRewards):
    """Reward terms for the MDP."""

    # Reward for keeping the hands close to the ball
    ball_close_to_hands = RewTerm(
        func=mdp.ball_close_to_hands_exp, weight=0.1, params={"std": 0.25}
    )

    # Reward for keeping the hands orientation consistent
    same_hands_orientation = RewTerm(
        func=mdp.same_hands_orientation_exp, weight=0.1, params={"std": 0.4}
    )


@configclass
class G1CatchBallRoughEnvCfg(CatchBallAndLocomotionVelocityRoughEnvCfg):
    rewards: G1CatchBallRewards = G1CatchBallRewards()

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
        self.rewards.joint_deviation_arms.weight = -0.05

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class G1OnlyWalkRoughEnvCfg(G1CatchBallRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # set the only walk rewards
        self.rewards: G1BaseWalkRewards = G1BaseWalkRewards()
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

        # remove ball from the scene
        self.scene.ball = None
        
        # remove ball-related settings
        self.observations.policy.ball_pos = ObsTerm(
            func=mdp.dummy_zero_obs,
            params={"dim": 3},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.policy.ball_vel = ObsTerm(
            func=mdp.dummy_zero_obs,
            params={"dim": 3},
            noise=Unoise(n_min=-10, n_max=10),  # TODO: check this noise size
        )
        self.events.ball_physics_material = None
        self.events.ball_physics_mass = None
        self.events.reset_ball = None
        self.terminations.ball_dropped = None


@configclass
class G1StandingCatchBallRoughEnvCfg(G1CatchBallRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # set null velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0, 0)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
