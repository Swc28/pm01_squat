# MyTask — PM01 站立 ↔ 深蹲强化学习任务

基于官方 `velocity` 包的**零侵入副本**：官方原包保持不动，本包独立演进，当前主要用于
自定义 PM01 人形机器人的**深蹲 / 站立动作**（关键帧跟踪，无需动作数据集）。
速度行走任务作为副线保留。

- 蹲起训练任务：`Isaac-Squat-MyTask-PM01-v0`
- 蹲起演示任务：`Isaac-Squat-MyTask-PM01-Play-v0`
- 速度训练/演示：`Isaac-Velocity-MyTask-PM01-v0` / `-Play-v0`
- 蹲起日志目录：`<仓库根>/logs/rsl_rl/mytask_pm01_squat/<时间戳>/`
- 速度日志目录：`<仓库根>/logs/rsl_rl/mytask_pm01/<时间戳>/`

## 一、主要任务：站立 ↔ 深蹲

### 1.1 动作原理

`mdp/commands.py` 的 `SquatPoseCommand` 以**余弦相位**在“站立姿态”和“深蹲关键帧”之间插值，
生成每步的 24 关节目标角 + 基座高度目标 + 相位，策略只需跟踪该命令：

```
phase = 0.5 * (1 - cos(2π·t / period_s))       # 0 = 站立，1 = 最深蹲
关节目标 = 站立默认角 + phase × (蹲姿关键帧 − 站立默认角)
基座高度 = 0.90 + phase × (0.795 − 0.90)       # 单位 m
```

- 周期 `period_s = 4.0`（2 s 蹲下 + 2 s 站起）
- 单局 `episode_length_s = 20.0` → 5 个完整蹲起周期
- 仿真 `sim.dt = 0.005`、`decimation = 4` → 策略控制频率 50 Hz

### 1.2 关节角度（当前 32° 蹲深）

| 关节 | 站立 | 深蹲 | 说明 |
|---|---|---|---|
| `J00/J06_HIP_PITCH_L/R` | 默认 | **-0.5585 rad（-32°）** | 大腿前倾 32° |
| `J03/J09_KNEE_PITCH_L/R` | 默认 | **+1.1170 rad（+64°）** | 小腿相对竖直 32° |
| `J04/J10_ANKLE_PITCH_L/R` | 默认 | **-0.5585 rad（-32°）** | 脚掌平放规则：`ankle = -(hip + knee)` |
| 肩 / 肘 / 腕 | 默认 | **不变** | 手臂全程固定，不参与蹲起（消除抽搐） |
| 髋 roll/yaw、踝 roll、腰、头 | 默认 | **不变** | 由 `joint_deviation_l1` 拉回默认 |

- 基座高度：站立 0.90 m → 最深蹲 **0.795 m**（几何自洽校验：
  `0.3358·cos32° + 0.3607·cos32° + 0.1435 + 0.0612 ≈ 0.795`）
- 脚掌平放由世界系脚掌朝向惩罚兜底，防止“垫脚”骗高度奖励

修改入口：

- 蹲多深 / 腿角关键帧：`mdp/commands.py` 的 `SquatPoseCommandCfg.squat_joint_pos`、`base_height_squat`
- 蹲多快：同文件 `period_s`；同步修改 `squat_env_cfg.py` 中 `base_height_velocity.period_s`
- 动作缩放 / 限幅 / Play 相机：`config/pm01/squat_env_cfg.py`

### 1.3 奖励设计（当前值）

自定义函数实现在 `mdp/rewards.py`，在 `squat_env_cfg.py` 的 `RewardsCfg` 中挂载：

| 奖励项 | 权重 | 关键参数 | 作用 |
|---|---|---|---|
| `termination_penalty` | -200.0 | — | 摔倒 / 基座触地终止惩罚 |
| `track_base_height` | 1.5 | std 0.05 | 跟踪相位插值的基座高度目标 |
| `base_height_velocity` | 2.0 | std 0.05、period 4 s、`height_delta=0.105` | 奖励竖直方向的升降过程 |
| `track_joint_pos` | 0.5 | std 0.4 | 全关节相位轨迹跟踪（辅助项） |
| `track_leg_joint_traj` | 2.0 | std 0.4 | 腿部 6 个 pitch 关节轨迹跟踪 |
| `torso_tilt` | -1.5 | `max_tilt_deg=25` | 躯干倾角超 25° 受罚 |
| `leg_angle_window` | -0.05 | 窗口 10°~38° | 限制大腿/小腿倾角窗口 |
| `foot_flat` | -0.5 | 脚 body 世界朝向 | 惩罚垫脚 / 脚跟抬起 |
| `flat_orientation_l2` | -1.0 | — | 躯干保持直立（防后仰退化） |
| `lin_vel_xy_l2` | -0.5 | — | 原地蹲起，不应平移 |
| `ang_vel_xy_l2` | -0.05 | — | 抑制横滚/俯仰角速度 |
| `joint_deviation_l1` | -0.5 | 髋 roll/yaw、踝 roll、**肩/肘全部**、腰、头 | 非蹲关节保持默认（手臂固定关键项） |
| `joint_torques_l2` | -2.0e-6 | — | 能耗 |
| `joint_acc_l2` | -1.0e-7 | — | 平滑 |
| `action_rate_l2` | -0.01 | — | 动作平滑 |

> 关节跟踪类奖励使用 `exp(-error/std²)` 核。**std 过小会导致 exp 饱和、奖励恒 0、没有梯度**
> （曾用腿 std=0.1 导致 3100 轮内该项始终为 0，训练发散），修改时务必先冒烟验证奖励非零。

### 1.4 终止条件

| 终止项 | 条件 |
|---|---|
| `time_out` | 到达 20 s 单局时长 |
| `base_contact` | `LINK_BASE` 接触力 > 1.0（摔倒） |
| `bad_orientation` | 躯干相对竖直倾角 > 0.7 rad（≈40°，允许类人前倾） |

### 1.5 动作配置

- 髋/膝/踝 pitch 的 action scale 覆盖为 **0.5**（默认 ≈0.23 不足以蹲到位）
- 关节位置目标限幅 `clip = {".*": (-1.5, 1.5)}`（rad，作用于 `scale·action + 默认角` 后的绝对目标），
  防止极端动作把物理状态打爆（曾导致 `value loss = inf`、`log_std = NaN` 崩溃）

## 二、PPO 超参数

`config/pm01/agents/rsl_rl_ppo_cfg.py` 中两套配置：

| 超参 | 蹲起任务（当前主用） | 速度任务 |
|---|---|---|
| `max_iterations` | **15000** | 5000 |
| `save_interval` | **3000** | 1000 |
| `num_steps_per_env` | 24 | 24 |
| `experiment_name` | `mytask_pm01_squat` | `mytask_pm01` |
| `init_noise_std` | **0.5** | 1.0 |
| `entropy_coef` | **0.003** | 0.01 |
| `learning_rate` | **5.0e-4**（adaptive） | 1.0e-3（adaptive） |
| `clip_param` / `desired_kl` | 0.2 / 0.01 | 0.2 / 0.01 |
| `gamma` / `lam` | 0.99 / 0.95 | 0.99 / 0.95 |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 | 5 / 4 |
| `max_grad_norm` | 1.0 | 1.0 |
| 网络结构 | actor/critic [512, 256, 128] ELU | 同左 |
| `empirical_normalization` | True | True |

15000 轮 + 每 3000 轮存点的产物：`model_0.pt`、`model_3000.pt`、`model_6000.pt`、
`model_9000.pt`、`model_12000.pt`，最终 `model_14999.pt`。

## 三、目录结构

```
mytask/
├── README.md                       # 本文件
├── squat_env_cfg.py                # 蹲起环境总配置：场景/命令/观测/奖励/终止
├── velocity_env_cfg.py             # 速度行走环境总配置（副线）
├── record_video.py                 # 蹲起离屏相机录视频脚本
├── mdp/
│   ├── commands.py                 # SquatPoseCommand：余弦相位关键帧插值
│   ├── rewards.py                  # 蹲起/速度自定义奖励函数
│   ├── terminations.py
│   └── curriculums.py
└── config/pm01/
    ├── __init__.py                 # gym.register() 注册 4 个任务
    ├── squat_env_cfg.py            # PM01 + 蹲起：动作 scale/clip、Play 相机
    ├── flat_env_cfg.py             # PM01 + 速度行走
    └── agents/
        └── rsl_rl_ppo_cfg.py       # 两套 PPO 超参
```

## 四、依赖与安装

| 依赖 | 版本要求 |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2（commit `c2277524`，版本不通用） |
| Python | 3.11（conda 环境 `robot`） |
| NVIDIA 驱动 | 580.x（595 会导致 RTX 渲染器段错误） |

```bash
conda activate robot
# 仓库为可编辑安装（pip install -e），本任务包改完即时生效；全新机器部署时执行：
pip install -e source/engineai_rl_lab
```

## 五、运行方式

所有命令**固定在仓库根目录执行**（日志路径与当前目录相关）：

```bash
cd /home/talos/engineai_rl_lab
conda activate robot
```

### 5.1 改参数后先冒烟验证（强烈建议）

```bash
python scripts/velocity/train.py --task Isaac-Squat-MyTask-PM01-v0 \
    --headless --num_envs 64 --max_iterations 500
```

打开 TensorBoard 确认 `Episode_Reward/track_leg_joint_traj` **明显非零**、
`Loss/value_function` 全程有限，再上正式训练。

### 5.2 正式训练

```bash
python scripts/velocity/train.py --task Isaac-Squat-MyTask-PM01-v0 \
    --headless --num_envs 4096
```

- `scripts/velocity/train.py` 是**通用入口脚本**，训练哪个任务只看 `--task`
- 轮次与保存间隔由 `PM01SquatPPORunnerCfg` 决定（15000 轮 / 每 3000 轮）
- 续训：追加 `--resume`（自动接最近一次 run），或 `--resume --load_run <时间戳> --checkpoint model_XXXX.pt`
- **奖励或关键帧改动后必须从头训**，不要 resume 旧 checkpoint

### 5.3 监控

```bash
python -m tensorboard.main --logdir logs/rsl_rl/mytask_pm01_squat
# 浏览器打开 http://localhost:6006
```

重点指标：

- `Metrics/squat_pose/error_joint_pos`、`error_base_height` 应持续下降
- `Episode_Reward/track_leg_joint_traj`、`track_joint_pos` 应显著大于 0
- `Loss/value_function` 必须有限（出现 inf 说明数值发散，需检查奖励与噪声）
- `Episode_Termination/base_contact` 越低越好

### 5.4 演示（查看训练效果）

```bash
# GUI 窗口演示（必须加 --enable_cameras：Play 配置内置 TiledCamera）
python scripts/velocity/play.py --task Isaac-Squat-MyTask-PM01-Play-v0 \
    --enable_cameras --num_envs 1 \
    --load_run <时间戳> --checkpoint model_XXXX.pt

# GUI 闪退/黑屏（无显示服务器）→ 用 Play 场景内的 TiledCamera 离屏录视频
python -m engineai_rl_lab.tasks.mytask.record_video \
    --task Isaac-Squat-MyTask-PM01-Play-v0 --num_envs 1 \
    --load_run <时间戳> --checkpoint model_XXXX.pt --video_length 500
# 输出：./videos/<任务>_<run>_<checkpoint>.mp4（50 Hz 实时回放）
```

## 六、常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `A camera was spawned without the --enable_cameras flag` | Play 配置含 TiledCamera，命令需加 `--enable_cameras` |
| 找不到 `model.pt` | 本仓库按轮次保存为 `model_XXXX.pt`，用编号最大的那个 |
| `RuntimeError: normal expects all elements of std >= 0.0` | 实际是 `log_std` 变 NaN：value loss 先爆成 inf → 策略权重被污染。见修改日志 2026-10-24 的修复，崩溃后用最近的 `model_XXXX.pt` 续训 |
| 训练很久但 `track_leg_joint_traj` 恒 0 | 奖励 std 过小导致 exp 饱和，需放宽 std（当前 0.4） |
| 手臂抽搐 | 手臂无命令目标时必须由 `joint_deviation_l1` 覆盖 `.*SHOULDER.*`/`.*ELBOW.*` |

## 七、副线：速度行走任务

当前行走目标：**0.5 m/s 直行 20 m（45 s 单局，留 5 s 余量），保持站立姿态、避免倾倒**。

| 项目 | 设置 | 位置 |
|---|---|---|
| 线速度指令 | `lin_vel_x=(0.5, 0.5)`，`lin_vel_y=(0, 0)` | `config/pm01/flat_env_cfg.py` |
| 转向指令 | `ang_vel_z=(0, 0)`、`heading_command=False`、`rel_standing_envs=0` | 同上 |
| 单局时长 | 训练/演示均为 45 s（0.5 m/s × 40 s = 20 m） | 同上 |
| 站立姿态 | `flat_orientation_l2=-2.0`、`base_height_l2=-1.5`（target 0.87 m） | 同上 |
| 防倾倒 | 终止 `base_contact`/`bad_orientation`/`root_height_below` + `push_robot` 抗扰（±0.3 m/s，8~12 s 一次） | 同上 |
| 动作安全 | `joint_pos.clip={".*": (-1.5, 1.5)}` rad | 同上 |

奖励设计（自定义函数见 `mdp/rewards.py`，挂载于 `velocity_env_cfg.py`）：

| 奖励项 | 权重 | 阈值 | 作用 |
|---|---|---|---|
| `track_lin_vel_xy_exp` | 1.5 | std 0.35 | 主任务：跟踪线速度指令 |
| `hip_extension_penalty` | -0.5 | 0.3 | 惩罚髋关节过度后伸 |
| `hip_flexion_reward` | +0.5 | 0.1~0.5 | 鼓励主动向前迈步 |
| `hip_symmetry_penalty` | -0.5 | 0.3 | 惩罚左右髋不对称 |
| `knee_bend_penalty` | -0.8 | 1.0 | 抑制膝盖过度弯曲 |
| `shoulder_flexion_penalty` | -0.5 | 0.3 | 抑制手臂摆动过大 |
| `feet_air_time` | 0.75 | 0.4 | 鼓励正常步态周期 |

```bash
# 训练
python scripts/velocity/train.py --task Isaac-Velocity-MyTask-PM01-v0 --headless --num_envs 4096
# 监控
python -m tensorboard.main --logdir logs/rsl_rl/mytask_pm01
# 演示
python scripts/velocity/play.py --task Isaac-Velocity-MyTask-PM01-Play-v0 \
    --num_envs 1 --load_run <时间戳> --checkpoint model_XXXX.pt
```

## 八、修改日志

| 日期 | 修改项 | 原值 | 新值 | 备注 |
|---|---|---|---|---|
| 2026-10-05 | weight | 1.0 | 1.5 | 速度任务奖励权重提升 |
| 2026-10-05 | std | 0.5 | 0.35 | 降低探索噪声 |
| 2026-10-06 | 新增蹲起任务 | — | `Isaac-Squat-MyTask-PM01-v0/-Play-v0` | 关键帧跟踪，无需数据集；新增 mdp/commands.py、squat_env_cfg.py、record_video.py |
| 2026-10-16 | 深蹲+摆臂 | 蹲姿 hip -18°/膝 36°/踝 -18°，base_height_squat 0.65，无摆臂；腿角窗口上限 30°；手臂纳入 joint_deviation 惩罚；PPO 10000 轮/每 500 保存 | 蹲姿 hip -35.8°/膝 68.8°/踝 -33°，base_height_squat 0.78，摆臂肩 -14.3°+肘 -20.1°；腿角窗口上限 40°；解除手臂偏差惩罚；PPO 20000 轮/每 5000 保存 | 深蹲 12cm+防倾倒摆臂；episode 20s=5 周期；10-18 发现被回退，同日重新应用 |
| 2026-10-18 | 蹲起 GUI 演示命令修正 | 无 --enable_cameras | GUI 演示必须加 --enable_cameras | Play 配置内置 TiledCamera |
| 2026-10-23 | 蹲起手臂固定 + 训练轮次 | 摆臂关键帧（肩 -17.2°/肘 -22.9°）+ `track_arm_joint_pose`(1.2) + 手臂仅约束 ROLL/YAW；PPO 20000 轮/每 5000 存 | 移除摆臂关键帧与手臂跟踪奖励，`.*SHOULDER.*`/`.*ELBOW.*` 全部纳入 `joint_deviation_l1`；PPO 15000 轮/每 5000 存 | 消除手臂抽搐；腿部 32° 蹲深不变，仅保留深蹲/站立腿部动作 |
| 2026-10-24 | 蹲起训练发散修复 | 关节跟踪 std：腿 0.1、全关节 0.2（奖励恒 0 无梯度）；`track_base_height` 3.0；`init_noise_std=1.0`、`entropy_coef=0.01`、`lr=1e-3`；无关节目标限幅 | 腿 std 0.4、全关节 std 0.4；`track_base_height` 1.5；`init_noise_std=0.5`、`entropy_coef=0.003`、`lr=5e-4`；新增 `joint_pos.clip={".*": (-1.5, 1.5)}` | 崩溃根因：腿部跟踪奖励恒 0 + 探索噪声过大 → return 溢出 → `log_std` NaN；当时 save_interval=5000，崩溃丢 3100+ 轮 |
| 2026-10-24 | 训练轮次与保存间隔调整 | 15000 轮 / 每 1000 轮 | 15000 轮 / **每 3000 轮** | 平衡磁盘占用与崩溃损失；最终产物 `model_14999.pt` |
| 2026-10-24 | 腿轨迹奖励权重回退 | `track_leg_joint_traj` 3.0 | **2.0** | 单独验证放宽 std 的影响，不改权重以隔离变量 |
| 2026-10-27 | 行走任务参数（0.5 m/s 直行 20 m） | 速度 `(0,1.0)`/侧移 ±0.3/转向 ±0.8/heading 开启/单局 20 s/`push_robot=None`；姿态权重 -1.0；PPO 3000 轮/每 500 存 | `lin_vel_x=(0.5,0.5)`、`lin_vel_y=0`、`ang_vel_z=0`、`heading_command=False`、`rel_standing_envs=0`、单局 45 s；`flat_orientation_l2=-2.0`、`base_height_l2=-1.5`、`joint_pos.clip=±1.5`；启用 `push_robot` ±0.3 m/s（8~12 s）；PPO 5000 轮/每 1000 存 | 仅改行走专属文件（`flat_env_cfg.py`、`PM01MyTaskPPORunnerCfg`）；蹲起配置与 `mdp/` 共享函数保持不变 |

## 九、维护约定

- 不修改官方 `velocity/`、`tracking/` 任务包，保留基线对照
- 每次参数修改在本 README 的“修改日志”中追加一行
- 注册名（`config/pm01/__init__.py`）、配置类名、`experiment_name` 三处必须一一对应
- 改奖励 / 关键帧后先跑 500 轮冒烟，确认奖励项非零再正式训练
