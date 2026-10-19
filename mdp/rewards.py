from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def both_feet_airborne_penalty(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    both_airborne = torch.sum(contacts.int(), dim=1) == 0
    return both_airborne.float()


def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    return torch.square(height - target_height)


def knee_bend_penalty(env, threshold: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    knee_joint_ids = asset.find_joints(".*KNEE_PITCH.*")[0]
    if len(knee_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    knee_pos = joint_pos[:, knee_joint_ids]
    penalty = torch.sum(torch.clamp(knee_pos - threshold, min=0.0), dim=1)
    return penalty


def arm_swing_reward(env, max_amplitude: float = 0.35, pos_weight: float = 0.6, vel_weight: float = 0.4, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel
    shoulder_joint_ids = asset.find_joints(".*SHOULDER_PITCH.*")[0]
    if len(shoulder_joint_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    left_pos = joint_pos[:, shoulder_joint_ids[0]]
    right_pos = joint_pos[:, shoulder_joint_ids[1]]
    left_vel = joint_vel[:, shoulder_joint_ids[0]]
    right_vel = joint_vel[:, shoulder_joint_ids[1]]
    # 1. 幅度限制：两边都不能超出范围（防止一手举头顶）
    amp_mask = (torch.abs(left_pos) < max_amplitude) & (torch.abs(right_pos) < max_amplitude)
    # 2. 位置反向：左右肩位置异号（一前一后）
    pos_reward = -left_pos * right_pos
    pos_reward = torch.clamp(pos_reward, min=0.0)
    # 3. 速度反向：左右肩速度异号（真正在摆动，不是摆好不动）
    vel_reward = -left_vel * right_vel
    vel_reward = torch.clamp(vel_reward, min=0.0)
    # 组合：必须满足幅度限制，同时鼓励位置+速度都反向
    reward = amp_mask.float() * (pos_weight * pos_reward + vel_weight * vel_reward)
    return reward


def hip_extension_penalty(env, threshold: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    hip_joint_ids = asset.find_joints(".*HIP_PITCH.*")[0]
    if len(hip_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    hip_pos = joint_pos[:, hip_joint_ids]
    penalty = torch.sum(torch.clamp(hip_pos - threshold, min=0.0), dim=1)
    return penalty


def shoulder_flexion_penalty(env, threshold: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    shoulder_joint_ids = asset.find_joints(".*SHOULDER_PITCH.*")[0]
    if len(shoulder_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    shoulder_pos = joint_pos[:, shoulder_joint_ids]
    penalty = torch.sum(torch.clamp(shoulder_pos - threshold, min=0.0), dim=1)
    return penalty


def hip_flexion_reward(env, min_angle: float = 0.1, max_angle: float = 0.6, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    hip_joint_ids = asset.find_joints(".*HIP_PITCH.*")[0]
    if len(hip_joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    hip_pos = joint_pos[:, hip_joint_ids]
    flex = -hip_pos
    reward = torch.sum(torch.clamp(flex - min_angle, min=0.0) * torch.clamp(max_angle - flex, min=0.0), dim=1)
    return reward


def hip_symmetry_penalty(env, threshold: float = 0.3, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    hip_joint_ids = asset.find_joints(".*HIP_PITCH.*")[0]
    if len(hip_joint_ids) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    left_hip = joint_pos[:, hip_joint_ids[0]]
    right_hip = joint_pos[:, hip_joint_ids[1]]
    diff = torch.abs(left_hip - right_hip)
    penalty = torch.clamp(diff - threshold, min=0.0)
    return penalty


def track_squat_joint_pos_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """跟踪 SquatPoseCommand 命令的关节角目标（命令向量的前 num_joints 维）。

    注意：asset_cfg 必须按 articulation 顺序覆盖全部关节（默认即可），
    否则与命令向量的关节顺序不对齐。
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    joint_targets = command[:, : asset.data.joint_pos.shape[1]]
    error = torch.sum(torch.square(joint_targets - asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-error / std**2)


def track_squat_base_height_exp(env, std: float, command_name: str) -> torch.Tensor:
    """跟踪 SquatPoseCommand 命令的基座高度目标（命令向量的倒数第 2 维）。"""
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    height_target = command[:, -2]
    error = torch.square(height_target - asset.data.root_pos_w[:, 2])
    return torch.exp(-error / std**2)


def lin_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """惩罚基座水平面线速度（原地蹲起不应平移）。"""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def leg_angle_window_penalty(
    env, min_deg: float = 10.0, max_deg: float = 30.0, command_name: str = "squat_pose"
) -> torch.Tensor:
    """约束深蹲时大腿/小腿倾角落在 [min_deg, max_deg] 窗口内。

    倾角定义（相对竖直方向，脚掌平放时）：
        大腿倾角 = |hip_pitch|；小腿倾角 = |hip_pitch + knee_pitch|。
    仅在蹲下相位激活（phase>0.5 线性门控，phase=1 即最深蹲时完全生效），
    站立相位不约束（站立时倾角 <10° 是正常的）。
    返回超出窗口的角度总和（度），奖励端取负权重即成惩罚。
    """
    asset = env.scene["robot"]
    hip_ids, _ = asset.find_joints(["J00_HIP_PITCH_L", "J06_HIP_PITCH_R"], preserve_order=True)
    knee_ids, _ = asset.find_joints(["J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R"], preserve_order=True)
    hip = asset.data.joint_pos[:, hip_ids]
    knee = asset.data.joint_pos[:, knee_ids]
    thigh_deg = torch.rad2deg(torch.abs(hip))
    shank_deg = torch.rad2deg(torch.abs(hip + knee))
    viol = (
        torch.clamp(min_deg - thigh_deg, min=0.0)
        + torch.clamp(thigh_deg - max_deg, min=0.0)
        + torch.clamp(min_deg - shank_deg, min=0.0)
        + torch.clamp(shank_deg - max_deg, min=0.0)
    )
    # 相位门控：只在深蹲附近生效；双腿违规角度求和降为 (num_envs,) 再乘门控
    phase = env.command_manager.get_command(command_name)[:, -1]
    gate = torch.clamp((phase - 0.5) / 0.5, min=0.0)
    return viol.sum(dim=1) * gate


def foot_flat_pitch_penalty(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ANKLE.*")
) -> torch.Tensor:
    """惩罚脚掌不平放（垫脚/脚跟抬起），是防止垫脚的核心信号。

    原理：垫脚 = 整只脚绕脚尖旋转，踝关节角可以保持不变，因此关节空间
    约束检测不到垫脚；必须用世界系脚掌朝向。取脚 body 的世界四元数，
    计算脚体局部 z 轴在世界系下偏离竖直方向的程度：tilt = 1 - cos(tilt角)，
    平放时 ≈0，垫脚 δ° 时增量 ≈ 1-cos(δ)。
    注：默认站立姿态下若脚体坐标系与竖直方向存在小的常数偏置（数度内），
    只影响常数项不影响策略优化方向。
    """
    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, 2, 4)
    up = torch.tensor([0.0, 0.0, 1.0], device=asset.device).expand_as(foot_quat_w[..., :3])
    foot_z_w = quat_apply(foot_quat_w.reshape(-1, 4), up.reshape(-1, 3)).reshape(foot_quat_w.shape[0], -1, 3)
    tilt = 1.0 - foot_z_w[..., 2]  # (N, num_feet)
    return tilt.mean(dim=1)


def base_height_velocity_reward(
    env, std: float = 0.05, period_s: float = 4.0, height_delta: float = 0.12, command_name: str = "squat_pose"
) -> torch.Tensor:
    """奖励基座 z 高度“上升/下降过程”本身，使竖直运动获得正向奖励（蹲起过程奖励）。

    期望 z 速度幅值由余弦相位解析给出：
        |dz/dt| = height_delta * (2*pi/period_s) * sqrt(phase*(1-phase))
    峰值出现在 phase=0.5（蹲起中段，速度最大），两个锚点相位（0/1）处为 0。
    只比较速度幅值（不判断方向），避免依赖相邻帧差分，故无状态、无方向误判。
    说明：period_s / height_delta 应与此任务的 SquatPoseCommandCfg 保持一致，
    不一致只会让奖励信号略有偏差，不会报错。
    """
    asset = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    num_joints = asset.data.joint_pos.shape[1]
    phase = command[:, num_joints + 1]  # 命令布局 [关节目标, 目标高度, 相位]
    z_vel = torch.abs(asset.data.root_lin_vel_b[:, 2])  # 机身系竖直速度
    expected = height_delta * (2.0 * math.pi / period_s) * torch.sqrt(
        torch.clamp(phase * (1.0 - phase), min=0.0)
    )
    return torch.exp(-torch.square(z_vel - expected) / std**2)


def track_leg_joint_traj_exp(
    env, std: float = 0.2, command_name: str = "squat_pose"
) -> torch.Tensor:
    """跟踪腿部 6 个 pitch 关节（髋/膝/踝 × 左右）的余弦相位轨迹。

    命令向量的关节段与 articulation 关节顺序一致（由 SquatPoseCommand 按
    robot.data.default_joint_pos 顺序构造），因此直接用关节索引在同一顺序下对齐。
    """
    asset = env.scene["robot"]
    # 顺序：左髋/左膝/左踝/右髋/右膝/右踝（与 find_joints 传入顺序一致）
    leg_names = [
        "J00_HIP_PITCH_L", "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L",
        "J06_HIP_PITCH_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R",
    ]
    leg_ids, _ = asset.find_joints(leg_names, preserve_order=True)
    command = env.command_manager.get_command(command_name)
    num_joints = asset.data.joint_pos.shape[1]
    targets = command[:, :num_joints][:, leg_ids]
    error = torch.sum(torch.square(targets - asset.data.joint_pos[:, leg_ids]), dim=1)
    return torch.exp(-error / std**2)


def track_arm_joint_pose_exp(
    env, std: float = 0.3, command_name: str = "squat_pose"
) -> torch.Tensor:
    """蹲起时手臂姿态约束：跟踪肩/肘 pitch 关键帧目标（摆臂防倾倒）。

    手臂目标由 squat_joint_pos 关键帧给出（下蹲前摆、站起回位），
    单独成项可避免被全关节平均项稀释。
    """
    asset = env.scene["robot"]
    arm_names = [
        "J13_SHOULDER_PITCH_L", "J16_ELBOW_PITCH_L",
        "J18_SHOULDER_PITCH_R", "J21_ELBOW_PITCH_R",
    ]
    arm_ids, _ = asset.find_joints(arm_names, preserve_order=True)
    command = env.command_manager.get_command(command_name)
    num_joints = asset.data.joint_pos.shape[1]
    targets = command[:, :num_joints][:, arm_ids]
    error = torch.sum(torch.square(targets - asset.data.joint_pos[:, arm_ids]), dim=1)
    return torch.exp(-error / std**2)


def torso_tilt_penalty(
    env, max_tilt_deg: float = 25.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """身体倾斜度要求：惩罚躯干相对竖直方向倾斜超过 max_tilt_deg 的部分（度）。

    用 root 机身 z 轴在世界系下的仰角衡量合成倾角（roll+pitch），站立与蹲下均生效
    （统一阈值，比相位门控更直接满足“倾斜度要求”）。
    注意：max_tilt_deg 必须大于蹲起时的类人前倾角（约 20~30°），
    否则会与“允许自然前倾”的设计相冲突。
    """
    asset = env.scene[asset_cfg.name]
    up_local = torch.tensor([0.0, 0.0, 1.0], device=asset.device).expand(env.num_envs, 3)
    up_w = quat_apply(asset.data.root_quat_w, up_local)      # (N, 3)
    cos_tilt = torch.clamp(up_w[:, 2], -1.0, 1.0)            # (N,)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt))           # (N,)
    return torch.clamp(tilt_deg - max_tilt_deg, min=0.0)      # (N,)
