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
