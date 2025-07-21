# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg, CommandTermCfg, RewardTermCfg, TerminationTermCfg, ObservationTermCfg
from isaaclab.managers.observation_manager import ObservationGroupCfg 

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg 

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg 
from isaaclab.sim import SimulationCfg, PhysxCfg 
import isaaclab.sim as sim_utils

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp_reach 
import isaaclab.envs.mdp as mdp 

from isaaclab.envs.mdp import commands_cfg as mdp_commands_cfg 


##
# Scene settings
##

@configclass
class ReachSceneCfg:
    """Scene configuration for the Franka robot."""
    num_envs: int = 16384
    env_spacing: float = 2.5
    replicate_physics: bool = False 
    filter_collisions: bool = True 

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,
                "panda_joint7": 0.785,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        ),
        actuators={
            "arm_actuator": ImplicitActuatorCfg( 
                joint_names_expr=["panda_joint[1-7]"], 
                effort_limit_sim=87.0, 
                velocity_limit_sim=2.175, 
                stiffness=80.0, 
                damping=4.0, 
            ),
            "gripper_actuator": ImplicitActuatorCfg( 
                joint_names_expr=["panda_finger_joint.*"], 
                effort_limit_sim=200.0, 
                velocity_limit_sim=0.2, 
                stiffness=2e3, 
                damping=1e2, 
            ),
        },
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    end_effector_target_pose: mdp_commands_cfg.UniformPoseCommandCfg = mdp_commands_cfg.UniformPoseCommandCfg(
        asset_name="robot", 
        body_name="panda_hand", 
        resampling_time_range=(5.0, 5.0), 
        ranges=mdp_commands_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7), 
            pos_y=(-0.2, 0.2),
            pos_z=(0.2, 0.5), 
            roll=(-math.pi, math.pi),
            pitch=(-math.pi, math.pi),
            yaw=(-math.pi, math.pi),
        ),
        debug_vis=True, 
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp_reach.JointPositionActionCfg = mdp_reach.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["panda_joint[1-7]"], 
        clip={"panda_joint[1-7]": (-1.0, 1.0)}, 
    )
    gripper_action: mdp_reach.JointPositionActionCfg = mdp_reach.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["panda_finger_joint.*"], 
        clip={"panda_finger_joint.*": (-1.0, 1.0)}, 
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    end_effector_position_tracking: RewardTermCfg = RewardTermCfg(
        func=mdp_reach.position_command_error,
        weight=-10.0,
        params={"command_name": "end_effector_target_pose", 
                "asset_cfg": SceneEntityCfg(name="robot", body_names="panda_hand")}, 
    )
    end_effector_orientation_tracking: RewardTermCfg = RewardTermCfg(
        func=mdp_reach.orientation_command_error,
        weight=-1.0,
        params={"command_name": "end_effector_target_pose", 
                "asset_cfg": SceneEntityCfg(name="robot", body_names="panda_hand")}, 
    )
    joint_vel_l2: RewardTermCfg = RewardTermCfg(
        func=mdp_reach.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )
    action_rate_l2: RewardTermCfg = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True) 


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg): 
        """Observations for policy."""

        joint_pos: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(name="robot")}) 
        joint_vel: ObservationTermCfg = ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg(name="robot")}) 
        
        ee_position: ObservationTermCfg = ObservationTermCfg(
            func=mdp.root_pos_w, 
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names="panda_hand")}
        )
        ee_orientation: ObservationTermCfg = ObservationTermCfg(
            func=mdp.root_quat_w, 
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names="panda_hand")}
        )
        pose_command: ObservationTermCfg = ObservationTermCfg(
            func=mdp.generated_commands, 
            params={"command_name": "end_effector_target_pose"}
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Environment configuration
##

@configclass
class FrankaPoseReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka Pose Reach environment."""

    # simulation
    simulation: SimulationCfg = SimulationCfg(
        dt=0.005,
        device="gpu", 
        physx=PhysxCfg( 
            solver_type="TGS",
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.001,
            friction_correlation_distance=0.0005,
            enable_ccd=False,
        ),
    )

    # scene
    scene: ReachSceneCfg = ReachSceneCfg() 

    scene_cfg: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=scene.num_envs, 
        env_spacing=scene.env_spacing, 
        replicate_physics=scene.replicate_physics, 
        filter_collisions=scene.filter_collisions, 
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="plane",
    )

    # rewards
    rewards: RewardsCfg = RewardsCfg()

    # terminations
    terminations: TerminationsCfg = TerminationsCfg()

    # actions
    actions: ActionsCfg = ActionsCfg() 

    # observations
    observations: ObservationsCfg = ObservationsCfg()

    # commands を追加
    commands: CommandsCfg = CommandsCfg()

    # curriculum
    curriculum: object = None

    def __post_init__(self):
        """Post initialization: Define default values for the environment."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        # viewer settings (optional, but good to set sensible defaults)
        self.viewer.eye = (3.5, 3.5, 3.5)


@configclass
class FrankaPoseReachEnvCfg_PLAY(FrankaPoseReachEnvCfg): # <-- FrankaPoseReachEnvCfg を継承
    def __post_init__(self):
        # 親クラスの __post_init__ を呼び出す
        super().__post_init__()
        # PLAY モードでは curriculum を無効にする
        self.curriculum = None
        # 環境数を減らすなど、デモンストレーション向けの設定変更もここで行える
        self.scene.num_envs = 1 # 例: PLAYモードでは環境数を1にする
        self.viewer.eye = (1.5, 1.5, 1.5) # 例: PLAYモードではカメラ視点を調整