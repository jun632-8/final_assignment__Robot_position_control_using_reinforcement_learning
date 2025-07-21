# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import isaaclab_tasks.manager_based.manipulation.franka_pose_reach_task.franka_pose_reach_env
from . import agents

##
# Register Gym environments.
##

# Isaac-LObject-Push-Franka-v0 の登録
gym.register(
    id="Isaac-Franka-Pose-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.franka_pose_reach_task.franka_pose_reach_env_cfg:FrankaPoseReachEnvCfg", 
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.franka_pose_reach_task.agents.franka_pose_reach_ppo_cfg:FrankaPoseReachPPORunnerCfg",
    },
)

# Isaac-Franka-Pose-Reach-Play-v0 の登録
gym.register(
    id="Isaac-Franka-Pose-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.franka_pose_reach_task.franka_pose_reach_env_cfg:FrankaPoseReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.franka_pose_reach_task.agents.franka_pose_reach_ppo_cfg:FrankaPoseReachPPORunnerCfg",
    },
)