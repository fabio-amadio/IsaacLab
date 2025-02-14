# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Catch-Ball Gym environments.
##

gym.register(
    id="Isaac-Catch-Ball-Rough-G1-Dof-23-Only-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1Dof23CatchBallRoughOnlyWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dof23CatchBallRoughOnlyWalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Catch-Ball-Rough-G1-Dof-23-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1Dof23CatchBallRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dof23CatchBallRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Catch-Ball-Flat-G1-Dof-23-Only-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1Dof23CatchBallFlatOnlyWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dof23CatchBallFlatOnlyWalkPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Catch-Ball-Flat-G1-Dof-23-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1Dof23CatchBallFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1Dof23CatchBallFlatPPORunnerCfg",
    },
)
