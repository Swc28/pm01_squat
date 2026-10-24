import numpy as np
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

from engineai_rl_lab.tasks.tracking.robots.pm01 import PM01_ACTION_SCALE, PM01_CYLINDER_CFG
from engineai_rl_lab.tasks.mytask.squat_env_cfg import SquatEnvCfg


def _lookat_quat_wxyz(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """计算相机 world 姿态四元数 (w,x,y,z)：从 cam_pos 看向 target，USD 相机沿局部 -z 视线。"""
    cam_pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    z = -forward
    x = np.cross(np.asarray(up, dtype=float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    tr = float(np.trace(R))
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w, xq, yq, zq = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, xq, yq, zq = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, xq, yq, zq = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, xq, yq, zq = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return (float(w), float(xq), float(yq), float(zq))


@configclass
class PM01MyTaskSquatEnvCfg(SquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = PM01_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 关键修正：默认腿部 scale(≈0.23) 下蹲到位需要超分布的动作值，
        # 蹲起任务把髋/膝/踝 pitch 关节的动作 scale 覆盖为 0.5
        # 注意：PM01_ACTION_SCALE 中踝关节键为宽匹配 ".*ANKLE.*"（含 PITCH+ROLL），
        # 直接覆盖该键，不能新增 ".*ANKLE_PITCH.*"（会与宽匹配冲突报 Multiple matches）
        self.actions.joint_pos.scale = {
            **PM01_ACTION_SCALE,
            ".*HIP_PITCH.*": 0.5,
            ".*KNEE_PITCH.*": 0.5,
            ".*ANKLE.*": 0.5,
        }
        # 关节位置目标限幅（单位：rad，作用于 scale*action+offset 后的绝对目标）：
        # 防极端动作把物理状态打爆（曾导致 value loss = inf、log_std = NaN 崩溃）。
        # 蹲姿目标最大约 1.12 rad（膝），±1.5 rad 不限制正常蹲起
        self.actions.joint_pos.clip = {".*": (-1.5, 1.5)}

        self.terminations.base_contact.params["sensor_cfg"].body_names = "LINK_BASE"


@configclass
class PM01MyTaskSquatEnvCfg_PLAY(PM01MyTaskSquatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 40 s = 10 个蹲起周期，便于观察
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False

        # 离屏相机（仅 Play 场景）：无显示器环境录视频用，RecordVideo 视口截图在
        # headless 下是黑帧，必须用传感器相机（RTX 离屏渲染）。
        # 注意：不能放进训练配置——训练时 4096 个环境各挂一个相机既浪费显存，
        # 还会强制训练命令加 --enable_cameras（否则报
        # "A camera was spawned without the --enable_cameras flag"）
        self.scene.view_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/ViewCam",
            offset=CameraCfg.OffsetCfg(
                convention="world",
                pos=(2.5, -1.2, 1.2),
                rot=_lookat_quat_wxyz((2.5, -1.2, 1.2), (0.0, 0.0, 0.75)),
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            width=640,
            height=360,
            update_period=0,
            debug_vis=False,
        )
