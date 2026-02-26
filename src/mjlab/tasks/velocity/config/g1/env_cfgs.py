"""Unitree G1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
import mjlab.terrains as terrain_gen
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_vision_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 rough terrain config with depth camera observation (vision policy).

  Replaces the height_scan actor observation with a flattened depth image
  from a forward-facing camera mounted at the pelvis.

  Camera orientation: quat=(0.5, 0.5, -0.5, -0.5) makes the camera look
  forward (+X in pelvis frame). In MuJoCo, cameras look in the -Z direction
  of their frame, so this rotation maps -Z_cam → +X_pelvis.
  """
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Obstacle terrain has more potential contacts than rough terrain.
  # 200 was too high (OOM at training scale); 64 is enough for obstacles.
  cfg.sim.nconmax = 64

  depth_camera = CameraSensorCfg(
    name="depth_sensor",
    parent_body="robot/pelvis",
    pos=(0.15, 0.0, 0.05),
    quat=(0.5, 0.5, -0.5, -0.5),  # forward-facing: -Z_cam → +X_pelvis
    width=30,
    height=53,
    fovy=58.0,
    data_types=("depth",),
    use_textures=False,
    use_shadows=False,
    clone_data=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_camera,)

  # Terminate if torso (upper body) contacts terrain/obstacle geometry.
  # Matches H1 vision approach: legs brushing past pillars is acceptable,
  # torso contact is a clear failure. Depth camera provides early warning
  # so the policy should avoid contact long before torso is reached.
  upper_body_contact_cfg = ContactSensorCfg(
    name="upper_body_terrain_contact",
    primary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (upper_body_contact_cfg,)
  cfg.terminations["upper_body_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "upper_body_terrain_contact"},
  )

  # Large termination penalty: makes obstacle collision catastrophic.
  # Go2 vision uses -200; this is the primary signal driving avoidance.
  cfg.rewards["termination_penalty"] = RewardTermCfg(
    func=mdp.termination_penalty,
    weight=-200.0,
  )
  # Prevent robot from thrashing in front of a wall when stuck.
  cfg.rewards["stand_still_penalty"] = RewardTermCfg(
    func=mdp.stand_still_penalty,
    params={"command_name": "twist"},
    weight=-1.0,
  )

  del cfg.observations["actor"].terms["height_scan"]
  cfg.observations["actor"].terms["depth"] = ObservationTermCfg(
    func=mdp.process_depth_image,
    params={"sensor_name": "depth_sensor"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )

  # Swap flat terrain (teaches nothing) for discrete obstacles, keeping all
  # other locomotion terrains from ROUGH_TERRAINS_CFG. Matches H1 vision.
  # Height fixed at 1.5m (always in frame, always blocks G1). Width (0.3→1.5m)
  # is the curriculum variable: narrow pillars → wide walls.
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_generator.sub_terrains = {
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.1,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.1,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 1.0),
      platform_width=2.0,
      border_width=0.25,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.1,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
      proportion=0.6,
      obstacle_height_mode="fixed",
      num_obstacles=10,
      obstacle_height_range=(1.5, 1.5),
      obstacle_width_range=(0.3, 1.5),
      platform_width=1.0,
      border_width=0.25,
    ),
  }

  return cfg
