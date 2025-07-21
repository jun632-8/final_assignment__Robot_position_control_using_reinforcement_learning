# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# env_cfg のインポートパスとクラス名を変更
from .franka_pose_reach_env_cfg import FrankaPoseReachEnvCfg 


class FrankaPoseReachEnv(ManagerBasedRLEnv): # クラス名を変更
    """Frankaロボットのエンドエフェクタ目標ポーズ到達タスクのためのカスタム環境."""

    def __init__(self, cfg: FrankaPoseReachEnvCfg, sim_device: str, headless: bool): 
        # 環境の基本設定
        super().__init__(cfg, sim_device, headless)

        # Franka Panda の関節インデックスを取得
        self.franka_panda_joint_indices = self.robot.joint_by_name_indices("panda_joint.*")

        # --- MDP タームのオフセット計算などのセットアップ ---

        self._initial_robot_joint_pos = self.robot.data.default_joint_pos[self.franka_panda_joint_indices]

    def _reset_idx(self, env_ids: torch.Tensor):
        """指定された環境IDをリセットする。"""
        # 親クラスのリセットを呼び出す
        super()._reset_idx(env_ids)

        # ロボットの初期姿勢をリセット
        joint_pos = self._initial_robot_joint_pos.unsqueeze(0).repeat(len(env_ids), 1)
        self.robot.set_joint_positions(joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocities(torch.zeros_like(joint_pos), env_ids=env_ids)


    def _pre_physics_step(self, actions: torch.Tensor):
        """物理ステップの前に実行される処理。"""


    def _post_physics_step(self, dt: float):
        """物理ステップの後に実行される処理。"""
        # 親クラスの post_physics_step を呼び出す
        super()._post_physics_step(dt)
