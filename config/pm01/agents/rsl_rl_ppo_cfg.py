from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PM01MyTaskPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # 行走任务：固定 0.5 m/s 直行；新增步态/摆臂参考奖励，需在原 5000 轮基础上继续适配
    max_iterations = 8000
    save_interval = 1000
    experiment_name = "mytask_pm01"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # 1.0 → 0.5：与蹲起任务一致，降低初期探索噪声，避免参考跟踪奖励长期为 0
        init_noise_std=0.5,
        noise_std_type="log",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01 → 0.003：抑制 log_std 膨胀（曾导致动作发散）
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 1e-3 → 5e-4：配合 adaptive 调度，降低新增奖励带来的数值发散风险
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PM01SquatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    # 每 3000 轮存点：崩溃时最多丢 3000 轮（曾因 save_interval=5000 崩溃丢 3100+ 轮）
    save_interval = 3000
    experiment_name = "mytask_pm01_squat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # 1.0→0.5：原探索噪声 ±1×scale 远超关节跟踪奖励容差，训练初期奖励恒 0
        init_noise_std=0.5,
        noise_std_type="log",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01→0.003：防止 log_std 持续增大（日志中熵已达 34+，≈4 rad）导致动作发散
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 1e-3→5e-4：配合 adaptive 调度降低数值发散风险
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
