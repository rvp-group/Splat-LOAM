import torch
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: np.array
    normals: np.array


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W
    return np.float32(Rt)


def depth_to_points(camera,
                    depth: torch.Tensor) -> torch.Tensor:
    """
    Computes backprojection from a camera and it's
    corresponding [1, H, W] depth map

    Returns an [3, H, W] tensor of points in world coordinates
    """
    W, H = camera.image_width, camera.image_height
    c2w = torch.linalg.inv(
        camera.world_view_transform.T
    )
    intrins = camera.projection_matrix[:3, :3].T
    intrins_i = torch.linalg.inv(intrins)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        indexing="xy"
    )
    points = torch.stack([
        grid_x - 0.5, grid_y - 0.5,
        torch.ones_like(grid_x)
    ], dim=-1)
    rays_d = points @ intrins_i.T
    c0 = torch.cos(rays_d[..., 0])
    c1 = torch.cos(rays_d[..., 1])
    s0 = torch.sin(rays_d[..., 0])
    s1 = torch.sin(rays_d[..., 1])
    rays = torch.stack([
        c0 * c1,
        s0 * c1,
        s1
    ], dim=-1)
    rays_d = rays @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depth.squeeze(0).unsqueeze(-1) * rays_d + rays_o
    return points.permute(2, 0, 1)


def depth_to_normal(camera,
                    depth: torch.Tensor) -> torch.Tensor:
    """
    Compute a normal map given a camera and its corresponding depth map
    via gradients.

    Returns a [3, H, W] normal map
    """
    points = depth_to_points(camera, depth)  # [3, H, W]
    res = torch.zeros((3, camera.image_height, camera.image_width),
                      dtype=torch.float32,
                      device=depth.device)
    dx = torch.cat([points[..., 2:, 1:-1] -
                    points[..., :-2, 1:-1]], dim=0)
    dy = torch.cat([points[..., 1:-1, 2:] -
                    points[..., 1:-1, :-2]], dim=1)
    normals = torch.nn.functional.normalize(
        torch.cross(dx, dy, dim=0), dim=0)
    res[..., 1:-1, 1:-1] = normals
    return res


def compute_depth_gradient(depth: torch.Tensor,
                           valid_mask: torch.Tensor) -> torch.Tensor:
    # use log of depth to highlight local changes in depth
    depth = torch.nan_to_num(torch.log(depth), nan=0, posinf=0, neginf=0)
    res = torch.zeros_like(depth)
    dx = torch.cat([depth[..., 2:, 1:-1] -
                    depth[..., :-2, 1:-1]], dim=0)
    dx_mask = valid_mask[..., 2:, 1:-1] & valid_mask[..., :-2, 1:-1]
    dx = dx * dx_mask
    dy = torch.cat([depth[..., 1:-1, 2:] -
                    depth[..., 1:-1, :-2]], dim=1)
    dy_mask = valid_mask[..., 1:-1, 2:] & valid_mask[..., 1:-1, :-2]
    dy = dy * dy_mask
    grad = torch.sqrt(dx ** 2 + dy ** 2)
    res[..., 1:-1, 1:-1] = grad
    return res
