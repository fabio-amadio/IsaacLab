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
    id="Isaac-Velocity-Rough-G1-Only-Walk-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1OnlyWalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1OnlyWalkRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-Standing-Catch-Ball-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1StandingCatchBallRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandingCatchBallRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-G1-Catch-Ball-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1CatchBallRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CatchBallRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Only-Walk-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1OnlyWalkFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1OnlyWalkFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Standing-Catch-Ball-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1StandingCatchBallFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandingCatchBallFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-G1-Catch-Ball-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1CatchBallFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CatchBallFlatPPORunnerCfg",
    },
)

# ##
# # Register Catch-Soft-Ball Gym environments.
# ##

# gym.register(
#     id="Isaac-Velocity-Rough-G1-Standing-Catch-Soft-Ball-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.soft_ball_rough_env_cfg:G1StandingCatchBallRoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandingCatchSoftBallRoughPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Rough-G1-Catch-Soft-Ball-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.soft_ball_rough_env_cfg:G1CatchBallRoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CatchSoftBallRoughPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Flat-G1-Standing-Catch-Soft-Ball-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.soft_ball_flat_env_cfg:G1StandingCatchBallFlatEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1StandingCatchSoftBallFlatPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Isaac-Velocity-Flat-G1-Catch-Soft-Ball-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.soft_ball_flat_env_cfg:G1CatchBallFlatEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CatchSoftBallFlatPPORunnerCfg",
#     },
# )
