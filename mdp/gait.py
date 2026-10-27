"""PM01 行走解析步态参考（仅行走任务使用，与蹲起任务无关）。

设计说明：
- 相位由 episode 时间推导（env.episode_length_buf * env.step_dt），左右腿相差 0.5 相位。
  相位不进入观测，保持观测维度 84 不变，从而可以继承已有行走策略检查点。
- 足端轨迹定义在髋关节矢状面内；几何量取实测值（/tmp/probe_pm01.py）：
  大腿 0.339 m、小腿 0.364 m。IK 基准用默认站姿（root≈0.90 m）髋→踝 0.6965 m，
  步态中心高度用 root=0.87 m 对应的 0.6665 m，与 base_height_l2 目标保持一致。
- 支撑期占 0.6 周期：脚相对机身从 +A 线性后移到 -A。A 由命令速度保证运动学一致：
  2A = v * stance_ratio * T，即支撑脚相对机身的速度恰为 -v。
- 摆动期占 0.4 周期：脚从 -A 摆到 +A，抬脚高度 lift * sin(pi * u)。
- 用平面 2-link 逆运动学解出大腿/小腿绝对倾角，再以“默认站姿的名义解”为基准
  换算成髋/膝/踝 pitch 增量，保证默认姿态下增量恒为 0。
- 手臂与腿共享同一相位：左腿前摆（phase≈0，左脚在最前）时左肩后摆（正）、
  右肩前摆（负）；肩 pitch 正方向 = 手臂向后（实测）。肘保持适度屈曲。
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# 实测几何常量
# ---------------------------------------------------------------------------
LEG_L1 = 0.339          # 髋 pitch 关节 → 膝
LEG_L2 = 0.364          # 膝 → 踝 pitch 关节
HIP_ANKLE_X0 = -0.0108  # 默认站姿：踝在髋关节后方 0.0108 m（x 向前为正）
HIP_ANKLE_Z_DEFAULT = 0.6965  # 默认站姿（root≈0.90 m）髋 → 踝竖直距离，用于 IK 基准标定
HIP_ANKLE_Z_STANCE = 0.6665   # 行走机身高度 root=0.87 m（base_height_l2 目标）对应的髋→踝竖直距离

# PM01 默认关节角（与 tracking/robots/pm01.py 中 init_state 一致）
HIP_PITCH_DEFAULT = -0.06
KNEE_PITCH_DEFAULT = 0.12
ANKLE_PITCH_DEFAULT = -0.06
ELBOW_PITCH_DEFAULT = -0.25

# 关节软限位（实测 factor=0.9，用于给参考目标做安全限幅）
HIP_PITCH_LIMIT = (-2.8618, 2.1638)
KNEE_PITCH_LIMIT = (-0.2121, 2.2541)
ANKLE_PITCH_LIMIT = (-0.6104, 0.6540)

STANCE_RATIO = 0.6
SWING_LIFT = 0.04

LEG_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J06_HIP_PITCH_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
]
ARM_JOINT_NAMES = [
    "J13_SHOULDER_PITCH_L",
    "J16_ELBOW_PITCH_L",
    "J18_SHOULDER_PITCH_R",
    "J21_ELBOW_PITCH_R",
]


# ---------------------------------------------------------------------------
# 逆运动学
# ---------------------------------------------------------------------------
def _ik_angles_scalar(x: float, z: float) -> tuple[float, float]:
    """标量版 2-link IK：返回大腿/小腿相对竖直方向的绝对倾角（前倾为正）。"""
    r = math.hypot(x, z)
    r = min(max(r, abs(LEG_L1 - LEG_L2) + 1e-6), LEG_L1 + LEG_L2 - 1e-6)
    alpha = math.atan2(x, z)
    cos_beta = (r * r + LEG_L1 * LEG_L1 - LEG_L2 * LEG_L2) / (2.0 * r * LEG_L1)
    cos_gamma = (r * r + LEG_L2 * LEG_L2 - LEG_L1 * LEG_L1) / (2.0 * r * LEG_L2)
    beta = math.acos(min(max(cos_beta, -1.0), 1.0))
    gamma = math.acos(min(max(cos_gamma, -1.0), 1.0))
    return alpha + beta, alpha - gamma


def _ik_angles(x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """张量版 2-link IK（膝朝前，脚掌通过踝补偿保持水平）。"""
    r = torch.sqrt(x * x + z * z).clamp(
        min=abs(LEG_L1 - LEG_L2) + 1e-6, max=LEG_L1 + LEG_L2 - 1e-6
    )
    alpha = torch.atan2(x, z)
    cos_beta = ((r * r + LEG_L1 * LEG_L1 - LEG_L2 * LEG_L2) / (2.0 * r * LEG_L1)).clamp(-1.0, 1.0)
    cos_gamma = ((r * r + LEG_L2 * LEG_L2 - LEG_L1 * LEG_L1) / (2.0 * r * LEG_L2)).clamp(-1.0, 1.0)
    return alpha + torch.acos(cos_beta), alpha - torch.acos(cos_gamma)


# 默认站姿名义解：所有增量以此为基准，默认姿态下增量恒为 0
_THETA_T_NOM, _THETA_S_NOM = _ik_angles_scalar(HIP_ANKLE_X0, HIP_ANKLE_Z_DEFAULT)


# ---------------------------------------------------------------------------
# 相位与足端轨迹
# ---------------------------------------------------------------------------
def gait_phase(env, period_s: float) -> torch.Tensor:
    """当前步态相位 [0, 1)，与 episode 时间同步。"""
    return torch.remainder(
        env.episode_length_buf.to(dtype=torch.float32) * env.step_dt / period_s, 1.0
    )


def foot_target(phase: torch.Tensor, stride: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor]:
    """相对名义足位的水平偏移 x 与抬脚高度 z（支撑期 z=0）。"""
    stance = phase < STANCE_RATIO
    u_st = (phase / STANCE_RATIO).clamp(0.0, 1.0)
    x_st = stride * (1.0 - 2.0 * u_st)
    u_sw = ((phase - STANCE_RATIO) / (1.0 - STANCE_RATIO)).clamp(0.0, 1.0)
    x_sw = -stride * torch.cos(math.pi * u_sw)
    z_sw = SWING_LIFT * torch.sin(math.pi * u_sw)
    zeros = torch.zeros_like(phase)
    return torch.where(stance, x_st, x_sw), torch.where(stance, zeros, z_sw)


# ---------------------------------------------------------------------------
# 关节参考目标
# ---------------------------------------------------------------------------
def leg_joint_targets(
    env, period_s: float = 0.9, command_name: str = "base_velocity"
) -> torch.Tensor:
    """腿部 6 个 pitch 关节目标角，顺序 [左髋, 左膝, 左踝, 右髋, 右膝, 右踝]。

    步幅由命令速度解析给出，保证支撑期脚相对机身速度与机身速度一致（2A = v·T_stance）。
    """
    command = env.command_manager.get_command(command_name)
    speed = command[:, 0].abs()
    stride = 0.5 * speed * STANCE_RATIO * period_s

    phase_l = gait_phase(env, period_s)
    phase_r = torch.remainder(phase_l + 0.5, 1.0)

    deltas = []
    for phase in (phase_l, phase_r):
        x, z = foot_target(phase, stride)
        theta_t, theta_s = _ik_angles(HIP_ANKLE_X0 + x, HIP_ANKLE_Z_STANCE - z)
        d_h = theta_t - _THETA_T_NOM
        d_s = theta_s - _THETA_S_NOM
        # 膝正方向 = 小腿相对大腿向后；踝补偿使脚掌保持水平
        deltas.extend([d_h, d_h - d_s, -d_s])

    deltas = torch.stack(deltas, dim=-1)
    defaults = torch.tensor(
        [HIP_PITCH_DEFAULT, KNEE_PITCH_DEFAULT, ANKLE_PITCH_DEFAULT] * 2,
        dtype=deltas.dtype,
        device=deltas.device,
    )
    hip_lo, hip_hi = HIP_PITCH_LIMIT
    knee_lo, knee_hi = KNEE_PITCH_LIMIT
    ankle_lo, ankle_hi = ANKLE_PITCH_LIMIT
    lower = torch.tensor([hip_lo, knee_lo, ankle_lo] * 2, dtype=deltas.dtype, device=deltas.device)
    upper = torch.tensor([hip_hi, knee_hi, ankle_hi] * 2, dtype=deltas.dtype, device=deltas.device)
    return torch.clamp(deltas + defaults, min=lower, max=upper)


def arm_joint_targets(
    env,
    period_s: float = 0.9,
    amplitude: float = 0.2,
    elbow_target: float = ELBOW_PITCH_DEFAULT - 0.05,
) -> torch.Tensor:
    """手臂 4 个 pitch 关节目标角，顺序 [左肩, 左肘, 右肩, 右肘]。

    与腿同相位反相摆动：左腿前摆时左肩后摆（+amplitude）、右肩前摆（-amplitude）；
    肘固定在略屈曲角度（默认 -0.25 rad，再屈 0.05 rad）。
    """
    phase = gait_phase(env, period_s)
    s = torch.cos(2.0 * math.pi * phase)
    shoulder_left = amplitude * s
    shoulder_right = -amplitude * s
    elbow = torch.full_like(s, elbow_target)
    return torch.stack([shoulder_left, elbow, shoulder_right, elbow], dim=-1)
