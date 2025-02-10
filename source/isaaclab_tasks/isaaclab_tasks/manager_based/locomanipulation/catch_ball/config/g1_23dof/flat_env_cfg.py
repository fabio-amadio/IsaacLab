# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import (
    G1Dof23CatchBallRoughEnvCfg,
    G1Dof23CatchBallRoughOnlyWalkEnvCfg,
    G1Dof23CatchBallRoughStandingEnvCfg,
)


@configclass
class G1Dof23CatchBallFlatEnvCfg(G1Dof23CatchBallRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.7
        self.rewards.different_step_times.weight = -1.0
        self.rewards.different_air_contact_times.weight = -1.0
        self.rewards.feet_swing_height.params["target_height"] = 0.18
        self.rewards.feet_swing_height.weight = -1.0
        self.rewards.feet_slide.weight = -0.5
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )


@configclass
class G1Dof23CatchBallFlatOnlyWalkEnvCfg(G1Dof23CatchBallRoughOnlyWalkEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.7
        self.rewards.different_step_times.weight = -1.0
        self.rewards.different_air_contact_times.weight = -1.0
        self.rewards.feet_swing_height.params["target_height"] = 0.18
        self.rewards.feet_swing_height.weight = -1.0
        self.rewards.feet_slide.weight = -0.5
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )


@configclass
class G1Dof23StandingCatchBallFlatEnvCfg(G1Dof23CatchBallRoughStandingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.7
        self.rewards.different_step_times.weight = -1.0
        self.rewards.different_air_contact_times.weight = -1.0
        self.rewards.feet_swing_height.params["target_height"] = 0.18
        self.rewards.feet_swing_height.weight = -1.0
        self.rewards.feet_slide.weight = -0.5
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
