# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the RSL-RL based PPO agent for the Franka Pose Reach task.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaPoseReachPPORunnerCfg(RslRlOnPolicyRunnerCfg): # クラス名を変更
    seed = 42 # シード値は必要に応じて変更
    num_steps_per_env = 24
    max_iterations = 1500 # 最大イテレーション数
    save_interval = 50
    experiment_name = "franka_pose_reach" # 実験名を変更
    run_name = ""
    resume = False
    empirical_normalization = False # これはRSL-RLの挙動によるので、必要であれば変更

    # PPO Policy 設定
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128], # 必要に応じて変更 (例: [64, 64])
        critic_hidden_dims=[512, 256, 128], # 必要に応じて変更 (例: [64, 64])
        activation="elu",
    )

    # PPO Algorithm 設定
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )