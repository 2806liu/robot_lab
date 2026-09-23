# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0


# 中文学习提示：这是 Unitree G1 崎岖地形速度跟踪任务。
# 相比 Flat，这里会涉及地形生成、地形课程和扰动随机化。初学时先确认
# Flat 任务稳定，再研究本文件中的地形和随机化配置。
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG  # isort: skip


@configclass
# 中文说明：该类是 G1 的任务专用覆盖层。父类提供通用 MDP，
# 本类替换 G1 资产、身体名称、观测缩放、动作缩放、奖励、随机化和命令范围。
class UnitreeG1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "torso_link"
    foot_link_name = ".*_ankle_roll_link"
    # fmt: off
    # joint_names = [
    #     "left_hip_pitch_joint",          # 0  L_LEG_HIP_PITCH
    #     "left_hip_roll_joint",           # 1  L_LEG_HIP_ROLL
    #     "left_hip_yaw_joint",            # 2  L_LEG_HIP_YAW
    #     "left_knee_joint",               # 3  L_LEG_KNEE
    #     "left_ankle_pitch_joint",        # 4  L_LEG_ANKLE_B
    #     "left_ankle_roll_joint",         # 5  L_LEG_ANKLE_A
    #     "right_hip_pitch_joint",         # 6  R_LEG_HIP_PITCH
    #     "right_hip_roll_joint",          # 7  R_LEG_HIP_ROLL
    #     "right_hip_yaw_joint",           # 8  R_LEG_HIP_YAW
    #     "right_knee_joint",              # 9  R_LEG_KNEE
    #     "right_ankle_pitch_joint",       # 10 R_LEG_ANKLE_B
    #     "right_ankle_roll_joint",        # 11 R_LEG_ANKLE_A
    #     "waist_yaw_joint",               # 12 WAIST_YAW
    #     "left_shoulder_pitch_joint",     # 15 L_SHOULDER_PITCH
    #     "left_shoulder_roll_joint",      # 16 L_SHOULDER_ROLL
    #     "left_shoulder_yaw_joint",       # 17 L_SHOULDER_YAW
    #     "left_elbow_pitch_joint",        # 18 L_ELBOW
    #     "left_elbow_roll_joint",         # 19 L_WRIST_ROLL
    #     "right_shoulder_pitch_joint",    # 22 R_SHOULDER_PITCH
    #     "right_shoulder_roll_joint",     # 23 R_SHOULDER_ROLL
    #     "right_shoulder_yaw_joint",      # 24 R_SHOULDER_YAW
    #     "right_elbow_pitch_joint",       # 25 R_ELBOW
    #     "right_elbow_roll_joint",        # 26 R_WRIST_ROLL
    # ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 中文说明：Scene 把通用场景中的机器人替换成 G1，并把高度扫描器
        # 绑定到 torso_link。所有 body/link 名称必须与 URDF 名称一致。
        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # 中文说明：观测是策略输入；scale 用于数值归一化，把项设为 None 会移除该观测。
        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # 中文说明：策略输出先经过关节位置动作项，再乘 action scale，成为关节目标偏移。
        # 跑步时动作幅度过大容易造成冲击和摔倒。
        # ------------------------------Actions------------------------------
        # reduce action scale
        # self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # self.actions.joint_pos.joint_names = self.joint_names

        # 中文说明：Events 是域随机化。startup 在创建环境时执行，reset 每回合执行。
        # 调试新奖励时可先缩小随机范围，确认基线后再恢复以提高 sim-to-real 鲁棒性。
        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # 中文说明：速度跟踪是主目标；姿态、动作变化、力矩、关节限制、脚滑和接触项
        # 约束运动质量。改跑步时重点检查 feet_air_time、feet_slide、速度跟踪和终止惩罚。
        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*hip_yaw.*", ".*hip_roll.*"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_arms_l1", -0.1, [".*shoulder.*", ".*elbow.*"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_torso_l1", -0.1, ["waist_yaw_joint"])
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still.weight = 0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle).*", "right_(hip|knee|ankle).*"]]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.action_mirror.weight = 0
        self.rewards.action_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle).*", "right_(hip|knee|ankle).*"]]

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Others
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeG1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # 中文说明：定义摔倒或其他失败何时结束回合。illegal_contact 通常用躯干碰地判断摔倒。
        # 终止过早会学不会，终止过晚会让策略持续积累无效数据。
        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # 中文说明：课程学习逐步增加地形或命令难度；当前命令课程被关闭，
        # 因此速度直接使用下面的固定范围。跑步实验可逐步恢复课程。
        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # 中文说明：lin_vel_x 是前后速度，lin_vel_y 是侧向速度，ang_vel_z 是旋转速度。
        # 初次改跑步建议只扩大 lin_vel_x，减少同时改变的变量。
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
