# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0


# 中文学习提示：这是 Unitree G1 平地速度跟踪任务的配置。
# 它在通用 velocity 环境上指定 G1 资产和 Flat 地面设置。建议先从该任务
# 复现稳定走路，再逐项修改目标速度和奖励，最后迁移到 Rough 或跑步任务。
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeG1RoughEnvCfg


@configclass
class UnitreeG1FlatEnvCfg(UnitreeG1RoughEnvCfg):
    # 中文说明：Flat 继承 Rough 的全部通用配置，再在这里覆盖平地设置。
    # 阅读顺序是先看父类，再看下面每一个 override，便于理解最终生效值。
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 中文说明：平地没有高低起伏，因此移除依赖地形高度的基座高度项。
        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # 中文说明：下面是平地专用奖励权重。正值鼓励目标行为，负值惩罚抖动、
        # 能耗或过大的关节力矩；权重的相对比例决定策略更重视什么。
        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]

        # Treat pelvis/torso ground contacts as failures in the flat locomotion task.
        # A small force is penalized first; a larger force terminates the episode.
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
        ]
        self.rewards.undesired_contacts.params["threshold"] = 0.1
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
        ]
        self.terminations.illegal_contact.params["threshold"] = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeG1FlatEnvCfg":
            self.disable_zero_weight_rewards()
