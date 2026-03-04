from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

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
  return depth.flatten(start_dim=1)  # [B, H*W]


def height_map_depth(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  voxel_size: float = 0.1,
  x_range: tuple[float, float] = (0.2, 2.2),
  y_range: tuple[float, float] = (-1.0, 1.0),
  z_range: tuple[float, float] = (-0.5, 2.0),
  max_depth: float = 4.0,
) -> torch.Tensor:
  """Convert a depth image to a 2.5D height map in the robot body frame.

  Back-projects each depth pixel to a 3D world point using the camera
  intrinsics and pose, transforms into the robot base frame, then bins
  points into a 2D (x, y) grid and records the maximum Z per cell.

  This is the depth-camera analogue of Go2 NaVILA's height_map_lidar:
  it avoids the sim-to-real domain gap of raw depth pixels fed to an MLP
  by producing a physically meaningful representation that is robust to
  rendering differences.

  Returns:
      [num_envs, nx * ny] where nx = round((x_range[1]-x_range[0])/voxel_size).
      Default grid is 20x20 = 400 values covering x=[0.2, 2.2]m, y=[-1, 1]m.
  """
  from mjlab.sensor import CameraSensor
  from mjlab.utils.lab_api.math import quat_apply_inverse

  sensor: CameraSensor = env.scene[sensor_name]
  depth = sensor.data.depth
  assert depth is not None
  depth = depth.squeeze(-1).float()  # [B, H, W]
  B, H, W = depth.shape
  device = depth.device

  # MuJoCo intrinsics: square pixels, principal point at image centre.
  fovy_deg = sensor.cfg.fovy if sensor.cfg.fovy is not None else 45.0
  fy = (H / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
  fx = fy
  cx, cy = W / 2.0, H / 2.0

  # Pixel grid [H, W].
  u = torch.arange(W, device=device, dtype=torch.float32)
  v = torch.arange(H, device=device, dtype=torch.float32)
  uu, vv = torch.meshgrid(u, v, indexing="xy")  # [H, W]

  # Valid depth mask.
  valid = torch.isfinite(depth) & (depth > 0.0) & (depth < max_depth)  # [B, H, W]

  # Back-project to camera frame.
  # MuJoCo convention: camera looks along -Z_cam; image v increases downward.
  X_cam = (uu - cx) / fx * depth   # [B, H, W]
  Y_cam = -(vv - cy) / fy * depth  # flip: image v↓ → camera Y↑
  Z_cam = -depth                   # depth is distance along -Z_cam

  N = H * W
  pts_cam = torch.stack([X_cam, Y_cam, Z_cam], dim=-1).reshape(B, N, 3)
  valid_flat = valid.reshape(B, N)

  # Camera world pose: cam_xpos [B,3], cam_xmat [B,9] → [B,3,3].
  cam_id = sensor.camera_idx
  cam_pos = env.sim.data.cam_xpos[:, cam_id]                    # [B, 3]
  cam_R = env.sim.data.cam_xmat[:, cam_id].reshape(B, 3, 3)     # [B, 3, 3]

  # Transform camera → world: P_world = R_cam @ P_cam + t_cam.
  pts_world = (cam_R @ pts_cam.transpose(1, 2)).transpose(1, 2) + cam_pos.unsqueeze(1)

  # Robot root pose.
  robot: Entity = env.scene["robot"]
  root_pos = robot.data.root_link_pos_w             # [B, 3]
  root_quat = robot.data.root_link_pose_w[:, 3:7]  # [B, 4] (w, x, y, z)

  # Transform world → robot body frame.
  pts_rel = pts_world - root_pos.unsqueeze(1)  # [B, N, 3]
  quat_exp = root_quat.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4)
  pts_body = quat_apply_inverse(quat_exp, pts_rel.reshape(B * N, 3)).reshape(B, N, 3)

  # Build 2D height map via vectorised scatter-max.
  nx = round((x_range[1] - x_range[0]) / voxel_size)
  ny = round((y_range[1] - y_range[0]) / voxel_size)
  n_cells = nx * ny

  px, py, pz = pts_body[..., 0], pts_body[..., 1], pts_body[..., 2]
  ix = ((px - x_range[0]) / voxel_size).long()
  iy = ((py - y_range[0]) / voxel_size).long()

  in_grid = (
    valid_flat
    & (ix >= 0) & (ix < nx)
    & (iy >= 0) & (iy < ny)
    & (pz >= z_range[0]) & (pz <= z_range[1])
  )  # [B, N]

  b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
  global_idx = b_idx * n_cells + ix * ny + iy  # [B, N]

  height_map = torch.full((B * n_cells,), z_range[0], device=device, dtype=torch.float32)
  height_map.scatter_reduce_(
    0, global_idx[in_grid], pz[in_grid], reduce="amax", include_self=True
  )
  return height_map.reshape(B, n_cells)  # [B, nx*ny]
