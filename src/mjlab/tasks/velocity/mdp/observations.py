from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))

def process_depth_image(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  near_clip: float = 0.3,
  far_clip: float = 2.0,
) -> torch.Tensor:
  """Process depth camera image for policy observation.

  Clamps depth values to [near_clip, far_clip], replaces NaN/Inf with
  far_clip, subtracts near_clip, and flattens to [num_envs, H*W].
  """
  from mjlab.sensor import CameraSensor

  sensor: CameraSensor = env.scene[sensor_name]
  depth = sensor.data.depth
  assert depth is not None
  depth = depth.clone().squeeze(-1)  # [B, H, W]
  depth[torch.isnan(depth)] = far_clip
  depth[torch.isinf(depth)] = far_clip
  depth = torch.clamp(depth, near_clip, far_clip) - near_clip
  return depth.reshape(env.num_envs, -1)  # [B, H*W]
