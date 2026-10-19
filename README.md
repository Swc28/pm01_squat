# MyTask — PM01 速度行走自定义训练任务

基于官方 `velocity` 任务的**零侵入副本**：官方原包保持不动，本包独立演进，
用于自定义奖励函数与训练参数的速度行走实验。

- 训练任务名：`Isaac-Velocity-MyTask-PM01-v0`
- 演示任务名：`Isaac-Velocity-MyTask-PM01-Play-v0`
- 训练日志目录：`<仓库根>/logs/rsl_rl/mytask_pm01/<时间戳>/`

## 项目功能

让 PM01 人形机器人在平地上学会跟踪速度指令行走（前进 + 转向），采用
Isaac Lab（ManagerBasedRLEnv）+ rsl_rl PPO 训练框架：

| 项目 | 训练配置 | 演示配置（Play） |
|---|---|---|
| 并行环境数 | 4096（OOM 可降 1024/512） | 50（演示可 `--num_envs 1` 近距离观察） |
| 线速度指令 x (m/s) | 0.3 ~ 1.0 | 固定 1.0 |
| 角速度指令 z (rad/s) | -0.8 ~ 0.8 | -0.5 ~ 0.5 |
| 单局时长 | 20 s | 40 s |
| 训练轮数 | 3000 轮，每 500 轮保存一次 checkpoint | — |

## 奖励设计（相对官方的主要自定义项）

自定义奖励函数实现位于 `mdp/rewards.py`，在 `velocity_env_cfg.py` 中挂载：

| 奖励项 | 权重 | 阈值 | 作用 |
|---|---|---|---|
| `track_lin_vel_xy_exp` | 1.5 | std 0.35 | 主任务：跟踪线速度指令（原 1.0/0.5） |
| `hip_extension_penalty` | -0.5 | 0.3 | 惩罚髋关节过度后伸（解决"脚勾在后面"） |
| `hip_flexion_reward` | +0.5 | 0.1~0.5 | 鼓励主动向前迈步 |
| `hip_symmetry_penalty` | -0.5 | 0.3 | 惩罚左右髋动作不对称 |
| `knee_bend_penalty` | -0.8 | 1.0 | 抑制膝盖过度弯曲 |
| `shoulder_flexion_penalty` | -0.5 | 0.3 | 抑制手臂摆动过大 |
| `feet_air_time` | 0.75 | 0.4 | 鼓励正常步态周期 |

其余塑形项（身体朝向、关节偏差、动作平滑、能耗等）与官方 velocity 一致，
完整清单见 `velocity_env_cfg.py` 的 `RewardsCfg`。

## 目录结构

```
mytask/
├── README.md                      # 本文件
├── velocity_env_cfg.py            # 环境总配置：场景/指令/观测/奖励/终止/课程
├── mdp/                           # 奖励、终止等函数实现
│   ├── rewards.py                 #   含 hip 系列、knee、shoulder 等自定义奖励
│   ├── terminations.py
│   └── curriculums.py
└── config/
    └── pm01/
        ├── __init__.py            # gym.register() 注册上面两个任务名
        ├── flat_env_cfg.py        # PM01MyTaskEnvCfg（训练）/ PM01MyTaskEnvCfg_PLAY（演示）
        └── agents/
            └── rsl_rl_ppo_cfg.py  # PPO 超参 + experiment_name="mytask_pm01"
```

## 依赖安装

| 依赖 | 版本要求 |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2（commit `c2277524`，版本不通用） |
| Python | 3.11（conda 环境 `robot`） |
| NVIDIA 驱动 | 580.x（595 会导致 RTX 渲染器段错误） |

```bash
# 1. 激活环境（自动注入 EXP_PATH 等 Isaac Sim 变量）
conda activate robot

# 2. 仓库为可编辑安装（pip install -e），新建任务包即时生效，无需重新安装
#    如为全新机器部署，在仓库根目录执行：
pip install -e source/engineai_rl_lab
```

## 运行方式

所有命令**固定在仓库根目录执行**（训练/演示的日志路径与当前目录相关）：

```bash
cd /home/talos/engineai_rl_lab
```

### 训练

```bash
python scripts/velocity/train.py --task Isaac-Velocity-MyTask-PM01-v0 \
    --headless --num_envs 4096
```

- `--headless`：训练无需渲染画面，省显存提速
- 续训：追加 `--resume`（自动接最近一次训练），或 `--resume --load_run <时间戳目录> --checkpoint model_XXXX.pt`
- checkpoint 保存位置：`logs/rsl_rl/mytask_pm01/<时间戳>/model_XXXX.pt`

### 监控训练

```bash
python -m tensorboard.main --logdir logs/rsl_rl/mytask_pm01
# 浏览器打开 http://localhost:6006
```

### 演示（查看训练效果）

```bash
# GUI 窗口（驱动 580 下可用）
python scripts/velocity/play.py --task Isaac-Velocity-MyTask-PM01-Play-v0 \
    --num_envs 1 --load_run <时间戳目录> --checkpoint model_XXXX.pt

# GUI 若崩溃 → headless 录视频（输出 <run>/videos/play/rl-video.mp4）
python scripts/velocity/play.py --task Isaac-Velocity-MyTask-PM01-Play-v0 \
    --num_envs 1 --load_run <时间戳目录> --checkpoint model_XXXX.pt \
    --headless --video --video_length 300
```

### 改参数后先小规模验证

```bash
python scripts/velocity/train.py --task Isaac-Velocity-MyTask-PM01-v0 \
    --headless --num_envs 64 --max_iterations 100
```

确认不报错、各奖励项非零后，再上 4096 环境正式训练。**奖励改了必须从头训**
（旧 checkpoint 是旧奖励下学出的策略，不要 `--resume`）。

## 蹲起任务（站立→蹲下→站立，关键帧跟踪，无需数据集）

原理：`mdp/commands.py` 的 `SquatPoseCommand` 按余弦相位在站立/蹲下关键帧间插值
（命令 = 24 关节目标 + 基座高度 + 相位），策略学习跟踪。蹲多深/多快改
`squat_env_cfg.py`（训练）、`config/pm01/squat_env_cfg.py`（scale/相机/Play）。

```bash
# 训练（默认 5000 轮，每 500 轮存 checkpoint，最终为 model_4999.pt）
python scripts/velocity/train.py --task Isaac-Squat-MyTask-PM01-v0 --headless --num_envs 4096

# 监控（重点看 Metrics/squat_pose/error_joint_pos、error_base_height 下降）
python -m tensorboard.main --logdir logs/rsl_rl/mytask_pm01_squat

# 演示（注意：本仓库不生成 model.pt，checkpoint 用编号最大的 model_XXXX.pt）
# 必须加 --enable_cameras：Play 配置内置 TiledCamera 离屏相机（录视频用），
# 不加会报 "A camera was spawned without the --enable_cameras flag"
python scripts/velocity/play.py --task Isaac-Squat-MyTask-PM01-Play-v0 \
    --enable_cameras --num_envs 1 --load_run <时间戳目录> --checkpoint model_XXXX.pt

# GUI 闪退/黑屏（无显示服务器）→ 离屏相机录视频（推荐）
# 官方 --video 走视口截图，headless 下是黑帧；本命令用 Play 场景内的
# TiledCamera 传感器离屏渲染逐帧写 mp4（依赖 imageio、imageio-ffmpeg）
python -m engineai_rl_lab.tasks.mytask.record_video \
    --task Isaac-Squat-MyTask-PM01-Play-v0 --num_envs 1 \
    --load_run <时间戳目录> --checkpoint model_XXXX.pt --video_length 500
# 输出：./videos/<任务>_<run>_<checkpoint>.mp4（50Hz 实时回放）
```

## 修改日志

| 日期 | 修改项 | 原值 | 新值 | 备注 |
|---|---|---|---|---|
| 2026-10-05 | weight | 1.0 | 1.5 | 奖励权重提升 |
| 2026-10-05 | std | 0.5 | 0.35 | 降低探索噪声 |
| 2026-10-06 | 新增蹲起任务 | — | `Isaac-Squat-MyTask-PM01-v0/-Play-v0` | 关键帧跟踪，无需数据集；新增 mdp/commands.py、squat_env_cfg.py、record_video.py |
| 2026-10-16 | 深蹲+摆臂 | 蹲姿 hip -18°/膝 36°/踝 -18°，base_height_squat 0.65，无摆臂；腿角窗口上限 30°；手臂纳入 joint_deviation 惩罚；PPO 10000 轮/每 500 保存 | 蹲姿 hip -35.8°/膝 68.8°/踝 -33°，base_height_squat 0.78，摆臂肩 -14.3°+肘 -20.1°；腿角窗口上限 40°；解除手臂偏差惩罚；PPO 20000 轮/每 5000 保存 | 深蹲 12cm+防倾倒摆臂（对齐 squat_motion CSV 参考动作）；episode 20s=5 周期不变；10-18 发现被回退，同日重新应用 |
| 2026-10-18 | 蹲起 GUI 演示命令修正 | 无 --enable_cameras | GUI 演示必须加 --enable_cameras | Play 配置内置 TiledCamera，不加报 "A camera was spawned without the --enable_cameras flag" |

## 维护约定

- 不修改官方 `velocity/`、`tracking/` 任务包，保留基线对照
- 每次参数修改在本 README 的"修改日志"中追加一行
- 注册名（`__init__.py`）、类名（`flat_env_cfg.py`、`rsl_rl_ppo_cfg.py`）、
  `experiment_name` 三处必须一一对应，改动前先互相核对
