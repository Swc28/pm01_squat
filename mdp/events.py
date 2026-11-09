"""行走任务：记录出生位姿，并在出生航向系下计算前向/侧向/航向误差。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _root_yaw(quat: torch.Tensor) -> torch.Tensor:
    """从 (w, x, y, z) 四元数取偏航角。"""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _as_env_index(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | slice | None):
    if env_ids is None:
        return slice(None)
    if isinstance(env_ids, slice):
        return env_ids
    if torch.is_tensor(env_ids):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def record_walk_spawn_pose(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """reset 之后记下根位置与航向，作为 20 m 直线的参考系。必须挂在 reset_base 之后。"""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_walk_spawn_pos_xy"):
        env._walk_spawn_pos_xy = torch.zeros(env.num_envs, 2, device=env.device)
        env._walk_spawn_yaw = torch.zeros(env.num_envs, device=env.device)
    idx = _as_env_index(env, env_ids)
    env._walk_spawn_pos_xy[idx] = asset.data.root_pos_w[idx, :2].clone()
    env._walk_spawn_yaw[idx] = _root_yaw(asset.data.root_quat_w[idx])


def walk_pose_in_spawn_frame(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """出生航向系下的前向位移、侧向位移、航向误差，形状均为 (N,)。"""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_walk_spawn_pos_xy"):
        record_walk_spawn_pose(env, None, asset_cfg)
    delta = asset.data.root_pos_w[:, :2] - env._walk_spawn_pos_xy
    yaw0 = env._walk_spawn_yaw
    cos0 = torch.cos(yaw0)
    sin0 = torch.sin(yaw0)
    x_fwd = cos0 * delta[:, 0] + sin0 * delta[:, 1]
    y_lat = -sin0 * delta[:, 0] + cos0 * delta[:, 1]
    yaw_err = _wrap_to_pi(_root_yaw(asset.data.root_quat_w) - yaw0)
    return x_fwd, y_lat, yaw_err


def walk_heading_error(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """观测：相对出生航向的误差 (N, 1)。"""
    return walk_pose_in_spawn_frame(env, asset_cfg)[2].unsqueeze(-1)


def walk_lateral_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """观测：出生航向系侧向位移 (N, 1)。"""
    return walk_pose_in_spawn_frame(env, asset_cfg)[1].unsqueeze(-1)


def heading_error_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚相对出生航向的偏航（未加权平方误差）。"""
    return torch.square(walk_pose_in_spawn_frame(env, asset_cfg)[2])


def lateral_drift_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚出生航向系下的侧向漂移。"""
    return torch.square(walk_pose_in_spawn_frame(env, asset_cfg)[1])


def forward_progress(
    env: ManagerBasedRLEnv,
    max_speed: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励沿出生航向的前向速度（截断到 [0, max_speed]），抑制原地踏步。"""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "_walk_spawn_yaw"):
        record_walk_spawn_pose(env, None, asset_cfg)
    yaw0 = env._walk_spawn_yaw
    vel_xy = asset.data.root_lin_vel_w[:, :2]
    v_fwd = torch.cos(yaw0) * vel_xy[:, 0] + torch.sin(yaw0) * vel_xy[:, 1]
    return torch.clamp(v_fwd, min=0.0, max=max_speed)
