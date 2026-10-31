import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import engineai_rl_lab.tasks.mytask.mdp as mdp
import torch
from engineai_rl_lab.tasks.mytask.mdp import gait as gait_mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


GAIT_PERIOD_S = 0.9


def _phase_sin(env):
    phase = gait_mdp.gait_phase(env, period_s=GAIT_PERIOD_S)
    return torch.sin(2.0 * math.pi * phase).unsqueeze(-1)


def _phase_cos(env):
    phase = gait_mdp.gait_phase(env, period_s=GAIT_PERIOD_S)
    return torch.cos(2.0 * math.pi * phase).unsqueeze(-1)


def _foot_contact(env, sensor_cfg: SceneEntityCfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    return (forces > 1.0).float()


def _foot_height(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2:3]


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",                  # ← 改平地
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LINK_BASE",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000.0, 1000.0),   # ← 不重采样，固定 0.5
        rel_standing_envs=0.0,                     # ← 不站桩
        rel_heading_envs=0.0,                      # ← 不随机朝向
        heading_command=False,                     # ← 关闭 heading
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),                  # ← 固定 0.5 m/s
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # ---- 新增：步态相位与足端感知 ----
        gait_phase_sin = ObsTerm(func=_phase_sin)
        gait_phase_cos = ObsTerm(func=_phase_cos)
        foot_contact = ObsTerm(
            func=_foot_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLE.*")},
        )
        foot_height = ObsTerm(
            func=_foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ANKLE.*")},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )

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
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLE.*"),
            "threshold": 0.25,                      # ← 0.4 → 0.25
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLE.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ANKLE.*"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HIP.*", ".*KNEE.*", ".*ANKLE.*"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HIP_YAW.*", ".*HIP_ROLL.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,                               # ← -0.1 → -0.05
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*SHOULDER.*", ".*ELBOW.*"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*WAIST.*", ".*HEAD.*"])}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)              # ← -0.5 → -0.1
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HIP.*", ".*KNEE.*", ".*ANKLE.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HIP.*", ".*KNEE.*"])},
    )
    both_feet_airborne = RewTerm(
        func=mdp.both_feet_airborne_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLE.*")},
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.3,                                # ← -1.0 → -0.3
        params={"target_height": 0.87},
    )
    knee_bend = RewTerm(
        func=mdp.knee_bend_penalty,
        weight=-0.5,
        params={"threshold": 1.2, "asset_cfg": SceneEntityCfg("robot")},
    )
    hip_extension = RewTerm(
        func=mdp.hip_extension_penalty,
        weight=-0.3,
        params={"threshold": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    gait_leg_tracking = RewTerm(
        func=mdp.track_gait_leg_joints_exp,
        weight=1.8,                                 # ← 1.2 → 1.8
        params={"std": 0.2, "period_s": 0.9, "command_name": "base_velocity"},   # ← std 0.5 → 0.2
    )
    gait_arm_tracking = RewTerm(
        func=mdp.track_gait_arm_joints_exp,
        weight=0.8,                                 # ← 0.5 → 0.8
        params={"std": 0.2, "period_s": 0.9, "command_name": "base_velocity"},   # ← std 0.4 → 0.2
    )
    # ---- 新增：摆动相抬脚高度 ----
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.5,
        params={
            "target_height": 0.08,
            "std": 0.03,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ANKLE.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ANKLE.*"),
        },
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="LINK_BASE"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.785},
    )
    root_height_below = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    )


@configclass
class CurriculumCfg:
    # 平地任务不需要地形 curriculum，先关闭；学好平地直走后可恢复
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    pass
@configclass
class VelocityEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 50.0   #← 20.0 → 50.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
