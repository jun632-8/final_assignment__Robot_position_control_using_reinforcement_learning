# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script is used to register custom environment to Isaac Lab."""

import os
import carb
from omni.isaac.lab.utils import register_gym_env_backend

# NOTE: The entry point must be a python module path.
# The string must be in the format: "<module_path>:<function_name>".
# For example: "my_extension.tasks.my_task:MyTask"
# The extension will import "my_extension.tasks.my_task" and then call the "MyTask" function.


ISAACLAB_TASKS_CUSTOM_ENV_CFG_ENTRY_POINT = {
    "Isaac-Franka-Pose-Reach-v0": "isaaclab_tasks.manager_based.manipulation.franka_pose_reach_task.franka_pose_reach_env_cfg:FrankaPoseReachEnvCfg",
}

@carb.log_exceptions
def carb_startup():
    """The carb_startup function that is called by omni.isaac.lab extension.

    This function registers the environment configurations to the Isaac Lab registry.
    """
    # Register the environment configurations to the Isaac Lab registry
    # We need to register the custom environment so that it can be used by the Isaac Lab
    # training scripts.
    for key, value in ISAACLAB_TASKS_CUSTOM_ENV_CFG_ENTRY_POINT.items():
        register_gym_env_backend(key, value)
        carb.log_info(f"Registered environment: {key} -> {value}")