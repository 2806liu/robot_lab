# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps before exiting.")
parser.add_argument("--video", action="store_true", help="Record a video from the zero-action rollout.")
parser.add_argument("--video_length", type=int, default=500, help="Number of frames to record.")
parser.add_argument(
    "--pure_stand",
    action="store_true",
    help="Disable reset/interval randomization and set all velocity commands to zero.",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Output folder for the video (default: logs/zero_agent/<task>/videos).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.pure_stand:
        # Isolate default-pose and PD stability from task randomization.
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        env_cfg.commands.base_velocity.rel_standing_envs = 1.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        for event_name in (
            "randomize_apply_external_force_torque",
            "randomize_reset_joints",
            "randomize_actuator_gains",
            "randomize_reset_base",
            "randomize_push_robot",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)
        if hasattr(env_cfg, "curriculum"):
            env_cfg.curriculum.command_levels_lin_vel = None
            env_cfg.curriculum.command_levels_ang_vel = None
        print("[INFO]: Pure-stand diagnostic enabled: zero commands and reset/interval randomization disabled.")
    # create environment; RGB rendering is required by RecordVideo.
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        task_name = args_cli.task.split(":")[-1]
        video_folder = args_cli.video_folder or os.path.join("logs", "zero_agent", task_name, "videos")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO]: Recording zero-action video to: {os.path.abspath(video_folder)}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate a finite rollout so a cloud task exits on its own.
    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    print(f"[INFO]: Completed zero-action rollout for {step + 1} steps.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
