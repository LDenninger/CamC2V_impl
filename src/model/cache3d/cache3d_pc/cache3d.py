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

from .utils import unproject_depth_map_to_point_map
from VidUtil.torch_utils.morphology import Dilation2d, Erosion2d
#from src.cache3d_pcd2.utils_np import unproject_depth_map_to_point_map


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
                 ):

        super().__init__()
        self._weight_dilation_steps = mask_dilation_steps
        self._weight_erosion_steps = mask_erosion_steps
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

                confidence_mask: Optional[torch.Tensor] = None, # TODO: support this
                resolution: Optional[Tuple[int, int]] = None,

                return_mask: bool = False,
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
        pts = rearrange(pts, '(B N) H W C -> B (N H W) C', B=images.shape[0])

        pts = pts.to(images.device, dtype=torch.float32)
        #pts = torch.from_numpy(pts).to(images.device, dtype=torch.float32)

        colors = rearrange(images, "B N C H W -> B (N H W) C")

        ## Render from pointclouds
        pts_list = [pts[b] for b in range(B)]
        cols_list = [colors[b] for b in range(B)]

        point_clouds = Pointclouds(
            points = pts_list,
            features = cols_list
        ).extend(T)

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

        #import ipdb; ipdb.set_trace()
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
