# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import KangarooRoughEnvCfg


@configclass
class KangarooFlatEnvCfg(KangarooRoughEnvCfg):
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

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # self.clip_actions = True

        # # 👇 打印每项 observation term 的维度（dim）
        # print("\n📦 Observation Term Dimensions in 'policy':")

        # for name, term in self.observations.policy.terms.items():
        #     print(f"{name:20s} dim: {term.dim}")

        # total_dim = sum(term.dim for term in self.observations.policy.terms.values())
        # print(f"\n🔷 Total observation dimension (policy input): {total_dim}")

class KangarooFlatEnvCfg_PLAY(KangarooFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot.interval_range_s = (9.5,10)
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        