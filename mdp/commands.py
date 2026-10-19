from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SquatPoseCommand(CommandTerm):
    """关键帧蹲起命令项：在站立姿态与蹲下姿态之间按余弦相位平滑插值。

    相位定义：``phase = 0.5 * (1 - cos(2*pi*t/period_s))``，每个周期内 0→1→0 循环，
    0 表示完全站立，1 表示最深蹲姿。

    命令向量（按最后一维）：
        ``[target_joint_pos (num_joints), target_base_height (1), phase (1)]``
    其中关节目标为绝对关节角（与 articulation 关节顺序一致）。

    注意：``period_s`` 应小于等于 episode 时长的一半，保证每局包含完整的蹲起周期。
    """

    cfg: SquatPoseCommandCfg

    def __init__(self, cfg: SquatPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # 站立姿态 = 机器人默认关节角（按 articulation 关节顺序）
        self._stand_joint_pos = self.robot.data.default_joint_pos[0].clone()
        # 蹲下姿态 = 站立姿态 + 配置的关键帧覆盖
        self._squat_joint_pos = self._stand_joint_pos.clone()
        if len(self.cfg.squat_joint_pos) > 0:
            joint_ids, _ = self.robot.find_joints(
                list(self.cfg.squat_joint_pos.keys()), preserve_order=True
            )
            self._squat_joint_pos[joint_ids] = torch.tensor(
                list(self.cfg.squat_joint_pos.values()), dtype=torch.float32, device=self.device
            )

        # 相位时钟（自 episode 重置起累计的秒数）
        self._phase_time = torch.zeros(self.num_envs, device=self.device)

        # metrics（自动以 Metrics/{命令名}/{项} 记入 TensorBoard）
        self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_base_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def phase(self) -> torch.Tensor:
        """蹲起相位，形状 (num_envs,)，范围 [0, 1]：0=站立，1=最深蹲姿。"""
        return 0.5 * (1.0 - torch.cos(2.0 * math.pi * self._phase_time / self.cfg.period_s))

    @property
    def command(self) -> torch.Tensor:
        phase = self.phase.unsqueeze(1)
        joint_pos_target = self._stand_joint_pos.unsqueeze(0) + phase * (
            self._squat_joint_pos - self._stand_joint_pos
        ).unsqueeze(0)
        base_height_target = self.cfg.base_height_stand + phase * (
            self.cfg.base_height_squat - self.cfg.base_height_stand
        )
        return torch.cat([joint_pos_target, base_height_target, phase], dim=1)

    def _update_command(self):
        # compute(dt) 不传 dt，用 env.step_dt 累加相位时钟
        self._phase_time += self._env.step_dt

    def _resample_command(self, env_ids: Sequence[int]):
        self._phase_time[env_ids] = 0.0

    def _update_metrics(self):
        command = self.command
        num_joints = self._stand_joint_pos.shape[0]
        self.metrics["phase"] = command[:, -1]
        self.metrics["error_joint_pos"] = torch.norm(
            command[:, :num_joints] - self.robot.data.joint_pos, dim=1
        )
        self.metrics["error_base_height"] = torch.abs(
            command[:, num_joints] - self.robot.data.root_pos_w[:, 2]
        )


@configclass
class SquatPoseCommandCfg(CommandTermCfg):
    """SquatPoseCommand 的配置项。"""

    class_type: type = SquatPoseCommand

    asset_name: str = "robot"
    """蹲起作用的目标机器人名称。"""

    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """不随时间重采样（周期相位由 _update_command 持续推进，仅 episode 重置时清零）。"""

    period_s: float = 4.0
    """单次蹲起周期（秒）：2 秒蹲下 + 2 秒站起。"""

    base_height_stand: float = 0.9
    """站立姿态目标基座高度（米）。"""

    base_height_squat: float = 0.78
    """最深蹲姿目标基座高度（米），需与 squat_joint_pos 的蹲深几何自洽。
    计算（等效腿长 L1=0.3358, L2=0.3607, 踝高 0.1435, 髋偏移 0.0612）：
    0.3358×cos(35.8°) + 0.3607×cos(33.0°) + 0.1435 + 0.0612 ≈ 0.780。
    注意：蹲深 35.8° 时踝背屈需求 = |hip| ≈ 35.8°，PM01 踝为 Q25 低刚度电机
    （刚度≈26.4，力矩52N·m）。若训练中出现垫脚（脚跟抬起/脚尖翘起），
    优先加大 foot_flat 惩罚权重；仍不行再回退蹲深（同步改 base_height_squat）。"""

    squat_joint_pos: dict[str, float] = {
        # 腿部深蹲（脚掌平放规则：ankle_pitch = -(hip_pitch + knee_pitch)）
        "J00_HIP_PITCH_L": -0.625,    # -35.8°：大腿倾角=|hip|=35.8°
        "J03_KNEE_PITCH_L": 1.201,    # +68.8°：小腿倾角=|hip+knee|=33.0°
        "J04_ANKLE_PITCH_L": -0.576,  # -(hip+knee)=-33.0°，脚掌平放
        "J06_HIP_PITCH_R": -0.625,
        "J09_KNEE_PITCH_R": 1.201,
        "J10_ANKLE_PITCH_R": -0.576,
        # 摆臂防倾倒：下蹲时手臂前摆（肩 pitch 负=前摆）提供前向角动量，
        # 抵消腿部前折的角动量；站起时回到默认姿态。肘部联动加屈。
        "J13_SHOULDER_PITCH_L": -0.25,  # 0 → -14.3°，前摆
        "J18_SHOULDER_PITCH_R": -0.25,
        "J16_ELBOW_PITCH_L": -0.35,     # -0.25 → -20.1°，前摆时肘多屈 5.7°
        "J21_ELBOW_PITCH_R": -0.35,
    }
    """蹲姿关键帧：关节名 → 目标角度（rad）。未列出的关节保持站立姿态。
    脚平放规则：ankle_pitch = -(hip_pitch + knee_pitch)。
    倾角定义：大腿 = |hip_pitch|，小腿 = |hip_pitch + knee_pitch|（相对竖直方向）。"""
