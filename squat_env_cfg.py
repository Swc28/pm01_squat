"""Squat keyframe tracking task env config: stand → squat → stand (no motion dataset).

与 velocity 任务同风格的零侵入配置：命令项每步生成相位插值的关节角/基座高度目标，
奖励引导策略原地完成周期性蹲起。
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import engineai_rl_lab.tasks.mytask.mdp as mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    """平地场景：无地形生成器、无高度扫描，仅接触传感器。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )


@configclass
class CommandsCfg:
    squat_pose = mdp.SquatPoseCommandCfg(
        asset_name="robot",
        period_s=4.0,
        base_height_stand=0.9,
        base_height_squat=0.6,  # 与 commands.py 深蹲关键帧几何自洽（腿倾角 35.8°/33°）
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        squat_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "squat_pose"})
        # 绝对关节角：与命令中的绝对关节目标直接对齐
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # ---- (1) 身高 z 上升/下降的大量奖励来源 ----
    # 1a. 跟踪相位插值的基座高度目标（权重最高）：z 越接近目标奖励越大
    track_base_height = RewTerm(
        func=mdp.track_squat_base_height_exp, weight=3.0, params={"command_name": "squat_pose", "std": 0.05}
    )
    # 1b. 奖励 z 高度“变化过程”本身：跟踪期望竖直速度幅值
    #     |dz/dt| = height_delta*(2π/period_s)*sqrt(phase*(1-phase))，中段最快、锚点为 0
    base_height_velocity = RewTerm(
        func=mdp.base_height_velocity_reward,
        weight=2.0,
        params={"command_name": "squat_pose", "period_s": 4.0, "height_delta": 0.30, "std": 0.05},
    )
    # 辅助：全关节相位轨迹跟踪（权重调低，避免稀释腿部/手臂专项项）
    track_joint_pos = RewTerm(
        func=mdp.track_squat_joint_pos_exp, weight=0.5, params={"command_name": "squat_pose", "std": 0.2}
    )
    # ---- (2) 腿按轨迹运动（髋/膝/踝 6 个 pitch 关节）----
    track_leg_joint_traj = RewTerm(
        func=mdp.track_leg_joint_traj_exp,
        weight=2.0,
        params={"command_name": "squat_pose", "std": 0.2},
    )
    # ---- (3) 蹲起时手臂约束（肩/肘 4 个 pitch 关节，摆臂防倾倒）----
    track_arm_joint_pose = RewTerm(
        func=mdp.track_arm_joint_pose_exp,
        weight=1.2,
        params={"command_name": "squat_pose", "std": 0.2},
    )
    # ---- (4) 身体倾斜度要求：躯干相对竖直方向倾角超过 25° 受罚 ----
    torso_tilt = RewTerm(func=mdp.torso_tilt_penalty, weight=-1.5, params={"max_tilt_deg": 32.0})
    # 蹲下相位限制大腿/小腿倾角窗口：上限 40° 覆盖深蹲关键帧（大腿 35.8°/小腿 33.0°）
    leg_angle_window = RewTerm(
        func=mdp.leg_angle_window_penalty,
        weight=-0.05,
        params={"min_deg": 10.0, "max_deg": 40.0, "command_name": "squat_pose"},
    )
    # 脚掌平放（防垫脚核心项）：世界系检测脚掌朝向，垫脚/脚跟抬起直接受罚
    foot_flat = RewTerm(
        func=mdp.foot_flat_pitch_penalty,
        weight=-05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ANKLE.*")},
    )
    # 躯干直立（类人）：权重调低允许蹲下时躯干自然前倾（人蹲起躯干前倾 20~30°，
    # 由髋屈曲代偿、降低踝背屈需求）；防真摔倒交给 bad_orientation 终止兜底
 """   flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)""" """取消惩罚"""
    lin_vel_xy_l2 = RewTerm(func=mdp.lin_vel_xy_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 非蹲关节保持默认姿态（髋roll/yaw、踝roll、腰、头）；
    # 手臂由摆臂命令驱动（防倾倒），不再惩罚偏差
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*HIP_ROLL.*",
                    ".*HIP_YAW.*",
                    ".*ANKLE_ROLL.*",
                    ".*WAIST.*",
                    ".*HEAD.*",
                ],
            )
        },
    )
    # 平滑与能耗
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="LINK_BASE"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},  # ~40°：允许类人前倾，只拦截真倾倒
    )


@configclass
class SquatEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        # 20 s = 5 个 4 s 蹲起周期
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
