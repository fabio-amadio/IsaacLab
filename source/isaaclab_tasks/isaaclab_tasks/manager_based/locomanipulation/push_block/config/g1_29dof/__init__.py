# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Push-Block Gym environments.
##

gym.register(
    id="Isaac-Push-Block-G1-Dof-29-Only-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_dof_29_env_cfg:G1Dof29PushBlockOnlyWalkCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dof29PushBlockOnlyWalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Push-Block-G1-Dof-29-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_dof_29_env_cfg:G1Dof29PushBlockEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dof29PushBlockPPORunnerCfg",
    },
)