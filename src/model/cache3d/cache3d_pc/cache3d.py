from typing import *
from pathlib import Path
import sys


import torch
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from pytorch3d.utils import cameras_from_opencv_projection


from einops import rearrange, repeat
from pytorch3d.structures import Pointclouds

from VidUtil.torch_utils.morphology import Dilation2d, Erosion2d
#from src.cache3d_pcd2.utils_np import unproject_depth_map_to_point_map

from VidUtil import Video
import torchvision

from typing import *

import numpy as np
import torch
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
from pytorch3d.utils import cameras_from_opencv_projection

from einops import rearrange, repeat
from pytorch3d.structures import Pointclouds

from VidUtil.torch_utils.morphology import Dilation2d, Erosion2d
#from src.cache3d_pcd2.utils_np import unproject_depth_map_to_point_map

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch


def unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        dm = depth_map[frame_idx]
        if dm.dim() == 3 and dm.size(-1) == 1:
            dm = dm.squeeze(-1)

        cur_world_points, _, _ = depth_to_world_coords_points(
            dm, extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)

    world_points_array = torch.stack(world_points_list, dim=0)
    return world_points_array


def depth_to_world_coords_points(
    depth_map: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        extrinsic (torch.Tensor): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: World coordinates (H, W, 3), camera coords (H, W, 3), and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None  # type: ignore[return-value]

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_se3 is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = torch.matmul(cam_coords_points, R_cam_to_world.T) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).

    Returns:
        torch.Tensor: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    # Intrinsic matrix must have zero skew (within tolerance)
    tol = 1e-8
    assert torch.abs(intrinsic[0, 1]) < tol and torch.abs(intrinsic[1, 0]) < tol, "Intrinsic matrix must have zero skew"

    device = depth_map.device
    dtype = depth_map.dtype

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates (match NumPy's default 'xy' behavior)
    u_vec = torch.arange(W, device=device, dtype=dtype)
    v_vec = torch.arange(H, device=device, dtype=dtype)
    u, v = torch.meshgrid(u_vec, v_vec, indexing='xy')  # both (H, W)

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(dtype)

    return cam_coords


def closed_form_inverse_se3(
    se3: torch.Tensor, R: torch.Tensor | None = None, T: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 torch.Tensor of SE3 matrices.
        R (optional): Nx3x3 torch.Tensor of rotation matrices.
        T (optional): Nx3x1 or Nx3 torch.Tensor of translation vectors.

    Returns:
        torch.Tensor: Inverted SE3 matrices with the same dtype/device as `se3`.

    Shapes:
        se3: (N, 4, 4) or (N, 3, 4)
        R: (N, 3, 3)
        T: (N, 3, 1) or (N, 3)
    """
    if se3.shape[-2:] not in [(4, 4), (3, 4)]:
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1) if se3 is (...,3,4); if (...,4,4) still works

    if T.dim() == 3 and T.size(-1) == 1:
        T = T  # (N,3,1)
    elif T.dim() == 2:
        T = T.unsqueeze(-1)  # (N,3) -> (N,3,1)
    else:
        raise ValueError(f"Unexpected T shape: {T.shape}")

    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)

    N = R_transposed.shape[0]
    inverted_matrix = torch.eye(4, device=se3.device, dtype=se3.dtype).unsqueeze(0).repeat(N, 1, 1)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


# TODO: this code can be further cleaned up


class Cache3DPCD(torch.nn.Module):

    def __init__(self,
                 resolution: Tuple[int, int] = (256, 256),
                 radius: float = 0.01,
                 points_per_pixel: int = 32,
                 bin_size: int | None = None,
                 mask_dilation_steps: int = 0,
                 mask_erosion_steps: int = 0,
                 mask_erosion_kernel_size: int = 3,
                 mask_dilation_kernel_size: int = 3,
                 return_mask: bool = False,
                 confidence_threshold: float = 0.0,
                 ):

        super().__init__()
        self._weight_dilation_steps = mask_dilation_steps
        self._weight_erosion_steps = mask_erosion_steps
        self.confidence_threshold = confidence_threshold
        self._resolution = resolution
        self._return_mask = return_mask

        raster_settings = PointsRasterizationSettings(
            image_size=resolution,
            radius=radius,           
            points_per_pixel=points_per_pixel,
            bin_size=bin_size               
        )

        self._renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=None, raster_settings=raster_settings),
            compositor=AlphaCompositor((0,0,0))
        )

        self._pointclouds = None

        if mask_erosion_steps > 0:
            self._mask_erosion = Erosion2d(kernel_size=mask_erosion_kernel_size)
        if mask_dilation_steps > 0:
            self._mask_dilation = Dilation2d(kernel_size=mask_dilation_kernel_size)



    @torch.no_grad()
    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self,
                images: torch.Tensor ,
                depths: torch.Tensor ,
                extrinsics: torch.Tensor,
                intrinsics: torch.Tensor,

                target_extrinsics: torch.Tensor,
                target_intrinsics: torch.Tensor,

                confidence: Optional[torch.Tensor] = None, # TODO: support this
                resolution: Optional[Tuple[int, int]] = None,

                return_mask: bool = False,
                save_pointcloud: bool = False,
                ):
        """
            Forward rendering.

            Arguments:
            :param images: [B, N, C, H, W] in [0, 1]
            :param depths: [B, N, 1, H, W] in meters
            :param extrinsics: [B, N, 4, 4] world to camera
            :param intrinsics: [B, N, 3, 3] camera intrinsics non-NDC
            :param target_extrinsics: [B, T, 4, 4] world to camera
            :param target_intrinsics: [B, T, 3, 3] camera intrinsics non-NDC

            Returns:
            :return: rendered images [B, T, C, H, W] in [0, 1]

        """
        #if isinstance(images, np.ndarray): images = torch.from_numpy(images)
        #if isinstance(depths, np.ndarray): depths = torch.from_numpy(depths)
        #if isinstance(extrinsics, np.ndarray): extrinsics = torch.from_numpy(extrinsics)
        #if isinstance(intrinsics, np.ndarray): intrinsics = torch.from_numpy(intrinsics)
        #if isinstance(target_extrinsics, np.ndarray): target_extrinsics = torch.from_numpy(target_extrinsics)
        #if isinstance(target_intrinsics, np.ndarray): target_intrinsics = torch.from_numpy(target_intrinsics)

        source_resolution = resolution or self._resolution
        device = images.device
        B, N = images.shape[:2]
        T = target_extrinsics.shape[1]
        ## Extract pointclouds
        depths = rearrange(depths, "B S 1 H W-> (B S) H W 1")
        extrinsics = rearrange(extrinsics, "B S ... -> (B S) ...")
        intrinsics = rearrange(intrinsics, "B S ... -> (B S) ...")
        # TODO: Change this to torch backend
        pts = unproject_depth_map_to_point_map(depths, extrinsics[:,:3], intrinsics)
        #pts = unproject_depth_map_to_point_map(depths.cpu().numpy(), extrinsics[:,:3].cpu().numpy(), intrinsics.cpu().numpy())
        pts = pts.to(images.device, dtype=torch.float32)
        #import ipdb; ipdb.set_trace()
        if confidence is not None:
            confidence = rearrange(confidence, "B S ... -> (B S) ...")
            confidence = torch.log(confidence + 1e-6)
            fmin = confidence.amin(dim=(1, 2, 3), keepdim=True)
            fmax = confidence.amax(dim=(1, 2, 3), keepdim=True)
            confidence = (confidence - fmin) / (fmax - fmin + 1e-6)
            confidence = confidence.squeeze(1)

            mask = confidence < self.confidence_threshold
            inval_t = -torch.ones((3,), device=pts.device)
            pts[mask] = inval_t

        #import ipdb; ipdb.set_trace()
        pts = rearrange(pts, '(B N) H W C -> B (N H W) C', B=images.shape[0])
        #if confidence is not None:
        #    masked_indices = (pts != inval_t).all(dim=-1)
        #    pts = pts[masked_indices]

        pts = pts.to(images.device, dtype=torch.float32)
        #pts = torch.from_numpy(pts).to(images.device, dtype=torch.float32)

        colors = rearrange(images, "B N C H W -> B (N H W) C")
        #if confidence is not None:
        #    colors = colors[masked_indices]

        ## Render from pointclouds
        pts_list = [pts[b] for b in range(B)]
        cols_list = [colors[b] for b in range(B)]

        point_clouds = Pointclouds(
            points = pts_list,
            features = cols_list
        )
        self._pointclouds = point_clouds if save_pointcloud else None
        point_clouds = point_clouds.extend(T)

        image_size = torch.Tensor([source_resolution]* (B*T)).to(device=device, dtype=torch.int64)
        R = target_extrinsics[:, :, :3, :3].reshape(B * T, 3, 3)
        t = target_extrinsics[:, :, :3, 3].reshape(B * T, 3)
        K = target_intrinsics.to(device).reshape(B * T, 3, 3)
        cameras = cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size)

        #self._renderer.rasterizer.cameras = cameras

        rendered = self._renderer(point_clouds, cameras=cameras)
        rendered = rearrange(rendered, "(B T) H W C -> B T H W C", B=B)
        #from VidUtil import Video
        #Video.fromArray(depths, "THWC").grid('horizontal', file="depth.png")
        #Video.fromArray(rendered[0], "THWC").grid(grid=(4,4), file="target.png")
        #Video.fromArray(images[0], "TCHW").grid('horizontal', file="source.png")

        return_mask = return_mask or self._return_mask        
        if return_mask:
            masks = (rendered != 0).all(dim=-1, keepdim=True).float()
            masks = self._process_mask(masks)

            return rendered, masks
        return rendered
    
    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def _process_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B, T, H, W, 1]
        B = mask.shape[0]
        mask = rearrange(mask, "B T H W C -> (B T) C H W")

        for _ in range(self._weight_dilation_steps):
            mask = self._mask_dilation(mask)
            mask = (mask >= 0.5).float()
        for _ in range(self._weight_erosion_steps):
            mask = self._mask_erosion(mask)
            mask = (mask >= 0.5).float()

        mask = rearrange(mask, "(B T) C H W -> B T H W C", B=B)
        return mask
    
    @property
    def pointclouds(self) -> Optional[Pointclouds]:
        return self._pointclouds
