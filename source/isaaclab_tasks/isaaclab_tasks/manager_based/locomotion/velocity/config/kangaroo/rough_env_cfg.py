# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ViewerCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import KANGAROO_CFG, KANGAROO_MINIMAL_CFG  # isort: skip
from isaaclab.terrains.config.rough import SIMPLE_ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class KangarooRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
    )

    # Penalize uneven step times between the two feets
    different_step_times = RewTerm(
        func=mdp.different_step_times,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
        },
    )

    # # Penalize too different feet air and contact times
    # different_air_contact_times = RewTerm(
    #     func=mdp.different_air_contact_times,
    #     weight=-0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
    #     },
    # )

    # both_feet_in_contact = RewTerm(
    #     func=mdp.both_feet_in_contact,
    #     weight=0.25,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
    #     },
    # )

    # # Penalize motor joint limits
    # dof_pos_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "leg_left_1_motor",
    #                 "leg_right_1_motor",
    #                 "leg_left_2_motor",
    #                 "leg_right_2_motor",
    #                 "leg_left_3_motor",
    #                 "leg_right_3_motor",
    #                 "leg_left_4_motor",
    #                 "leg_right_4_motor",
    #                 "leg_left_5_motor",
    #                 "leg_right_5_motor",
    #                 "leg_left_length_motor",
    #                 "leg_right_length_motor",
    #             ],
    #         )
    #     },
    # )

    # # Penalize deviation from default of the joints that are not essential for locomotion
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "leg_left_2_motor",
    #                 "leg_left_3_motor",
    #                 "leg_right_2_motor",
    #                 "leg_right_3_motor",
    #             ],
    #         )
    #     },
    # )

    # # Penalize base height distance from a given target
    # base_height_l2 = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-1.0,
    #     params={
    #         "target_height": 0.80,
    #     },
    # )

    # # Penalize distance from a target swing feet height
    # feet_swing_height = RewTerm(
    #     func=mdp.feet_swing_height,
    #     weight=-1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "target_height": 0.15,
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=".*_ankle_roll", preserve_order=True
    #         ),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
    #     },
    # )


@configclass
class KangarooActionsCfg:
    """Kangaroo Action specifications for the MDP."""

    joint_pos = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*_motor"],
        scale=0.1,
        use_tanh=True,
        clamp_offset=[0, 0, 0, 0, 0, 0, 0, 0, -0.54, -0.54, 0, 0],
    )
    # joint_vel = mdp.JointVelocityActionCfg(
    #     asset_name="robot", joint_names=[".*_motor"], scale=1.0
    # )


@configclass
class KangarooObservationsCfg:
    """Kangaroo Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        motor_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_motor",  # joint_idx: 4
                        "leg_right_1_motor",  # joint_idx: 11
                        "leg_left_2_motor",  # joint_idx: 18
                        "leg_left_3_motor",  # joint_idx: 19
                        "leg_right_2_motor",  # joint_idx: 36
                        "leg_right_3_motor",  # joint_idx: 37
                        "leg_left_4_motor",  # joint_idx: 38
                        "leg_left_5_motor",  # joint_idx: 40
                        "leg_left_length_motor",  # joint_idx: 43
                        "leg_right_length_motor",  # joint_idx: 45
                        "leg_right_4_motor",  # joint_idx: 46
                        "leg_right_5_motor",  # joint_idx: 48
                    ],
                )
            },
            noise=Unoise(n_min=-0.0025, n_max=0.0025),
        )
        motor_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_motor",  # joint_idx: 4
                        "leg_right_1_motor",  # joint_idx: 11
                        "leg_left_2_motor",  # joint_idx: 18
                        "leg_left_3_motor",  # joint_idx: 19
                        "leg_right_2_motor",  # joint_idx: 36
                        "leg_right_3_motor",  # joint_idx: 37
                        "leg_left_4_motor",  # joint_idx: 38
                        "leg_left_5_motor",  # joint_idx: 40
                        "leg_left_length_motor",  # joint_idx: 43
                        "leg_right_length_motor",  # joint_idx: 45
                        "leg_right_4_motor",  # joint_idx: 46
                        "leg_right_5_motor",  # joint_idx: 48
                    ],
                )
            },
            noise=Unoise(n_min=-0.025, n_max=0.025),
        )
        measured_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_joint",  # joint_idx: 1
                        "leg_right_1_joint",  # joint_idx: 2
                        "leg_left_2_joint",  # joint_idx: 7
                        "leg_right_2_joint",  # joint_idx: 8
                        "leg_left_3_joint",  # joint_idx: 14
                        "leg_right_3_joint",  # joint_idx: 15
                        "left_ankle_4_pendulum_joint",  # joint_idx: 21
                        "left_ankle_5_pendulum_joint",  # joint_idx: 23
                        "right_ankle_4_pendulum_joint",  # joint_idx: 30
                        "right_ankle_5_pendulum_joint",  # joint_idx: 32
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        measured_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "leg_left_1_joint",  # joint_idx: 1
                        "leg_right_1_joint",  # joint_idx: 2
                        "leg_left_2_joint",  # joint_idx: 7
                        "leg_right_2_joint",  # joint_idx: 8
                        "leg_left_3_joint",  # joint_idx: 14
                        "leg_right_3_joint",  # joint_idx: 15
                        "left_ankle_4_pendulum_joint",  # joint_idx: 21
                        "left_ankle_5_pendulum_joint",  # joint_idx: 23
                        "right_ankle_4_pendulum_joint",  # joint_idx: 30
                        "right_ankle_5_pendulum_joint",  # joint_idx: 32
                    ],
                )
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class KangarooEventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.4, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
    #         "mass_distribution_params": (0.0, 3.0),
    #         "operation": "add",
    #     },
    # )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0, 0),
                "y": (-0, 0),
                "z": (-0, 0),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (-0, 0),
            },
        },
    )
    # reset_motor_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.005, 0.005),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "leg_left_1_motor",
    #                 "leg_right_1_motor",
    #                 "leg_left_2_motor",
    #                 "leg_left_3_motor",
    #                 "leg_right_2_motor",
    #                 "leg_right_3_motor",
    #                 "leg_left_4_motor",
    #                 "leg_left_5_motor",
    #                 "leg_left_length_motor",
    #                 "leg_right_length_motor",
    #                 "leg_right_4_motor",
    #                 "leg_right_5_motor",
    #             ],
    #         ),
    #     },
    # )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #             "pitch": (-0.05, 0.05),
    #             "roll": (-0.05, 0.05),
    #         }
    #     },
    # )


@configclass
class KangarooTerminationsCfg:
    """Kangaroo Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    falling = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class KangarooCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # adjust_feer_air_time_weight = CurrTerm(
    #     func=mdp.adjust_feer_air_time_weight,
    #     params={
    #         "term_name": "feet_air_time",
    #         "weight": 1.0,
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
    #         "time_threshold": 0.6,
    #         "rew_threshold": 0.1,
    #     },
    # )
    # adjust_feer_air_time_weight = CurrTerm(
    #     func=mdp.adjust_feer_air_time_weight_alt,
    #     params={
    #         "term_name": "feet_air_time",
    #         "weight": 1.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
    #         "threshold": 0.1,
    #     },
    # )


@configclass
class KangarooViewerCfg(ViewerCfg):
    # HD: 1280 x 720; Full HD: 1920 x 1080; 4K: 3840 x 2160
    resolution: tuple[int, int] = (1920, 1080)
    # Default: (7.5, 7.5, 7.5) 
    eye: tuple[float, float, float] = (10, 10, 10) 


@configclass
class KangarooRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KangarooRewards = KangarooRewards()
    actions: KangarooActionsCfg = KangarooActionsCfg()
    observations: KangarooObservationsCfg = KangarooObservationsCfg()
    terminations: KangarooTerminationsCfg = KangarooTerminationsCfg()
    events: KangarooEventCfg = KangarooEventCfg()
    curriculum: KangarooCurriculumCfg = KangarooCurriculumCfg()
    viewer: KangarooViewerCfg = KangarooViewerCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.num_envs = 2048
        self.scene.robot = KANGAROO_MINIMAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso"
        self.scene.terrain.terrain_generator = SIMPLE_ROUGH_TERRAINS_CFG

        self.commands.base_velocity.debug_vis = False
        # self.commands.base_velocity.rel_standing_envs = 0.0

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Randomization
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }

        # Rewards
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.01

        self.rewards.dof_pos_limits = None
        # self.rewards.dof_pos_limits.weight = -0.1
        # self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
        #     "robot",
        #     joint_names=[".*_motor"],
        # )

        # self.rewards.flat_orientation_l2 = None
        self.rewards.flat_orientation_l2.weight = -2.0

        self.rewards.dof_acc_l2 = None
        # self.rewards.dof_acc_l2.weight = -1e-9
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*"]
        # )

        self.rewards.dof_torques_l2 = None
        # self.rewards.dof_torques_l2.weight = -1e-9
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*"]
        # )

        self.rewards.undesired_contacts = None
        # self.rewards.undesired_contacts.weight = -0.5
        # self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        #     "contact_forces", body_names=".*_ankle_roll"
        # )
        # self.rewards.undesired_contacts.params["threshold"] = 600.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class KangarooRoughEnvCfg_PLAY(KangarooRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
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
