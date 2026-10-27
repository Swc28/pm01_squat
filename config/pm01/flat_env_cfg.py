from isaaclab.utils import configclass

from engineai_rl_lab.tasks.tracking.robots.pm01 import PM01_ACTION_SCALE, PM01_CYLINDER_CFG
from engineai_rl_lab.tasks.mytask.velocity_env_cfg import VelocityEnvCfg


@configclass
class PM01MyTaskEnvCfg(VelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PM01_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.actions.joint_pos.scale = PM01_ACTION_SCALE
        # 关节位置目标限幅（rad，绝对目标）：防极端动作把物理状态打爆
        self.actions.joint_pos.clip = {".*": (-1.5, 1.5)}

        # 抗扰：周期性施加小幅推力，训练"避免倾倒"的鲁棒性
        self.events.push_robot.params = {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["LINK_BASE"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*ANKLE.*"
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*ANKLE.*"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*ANKLE.*"
        # 髋/膝/踝全腿限位惩罚（摆动期膝接近伸直，仅踝不足）
        self.rewards.dof_pos_limits.params["asset_cfg"].joint_names = [".*HIP.*", ".*KNEE.*", ".*ANKLE.*"]
        self.rewards.termination_penalty.weight = -200.0
        # 强化站立姿态：躯干竖直 + 保持基座高度
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.base_height_l2.weight = -1.5
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7

        # 行走任务：0.5 m/s 直行 20 m（0.5 m/s × 40 s），不转向、不侧移
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        # 20 m ÷ 0.5 m/s = 40 s，留 5 s 余量
        self.episode_length_s = 45.0

        self.terminations.base_contact.params["sensor_cfg"].body_names = "LINK_BASE"


@configclass
class PM01MyTaskEnvCfg_PLAY(PM01MyTaskEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 45.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # 演示：固定 0.5 m/s 直行，45 s 可走完 20 m
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
