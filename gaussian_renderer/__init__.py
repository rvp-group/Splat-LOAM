import torch
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.graphic_utils import depth_to_normal
from diff_surfel_spherical_rasterization import (
    GaussianRasterizer,
    GaussianRasterizationSettings
)


def render(
        camera: Camera,
        model: GaussianModel,
        depth_ratio: float = 0.0):

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.projection_matrix,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = torch.zeros_like(
        model.get_xyz, dtype=torch.float32,
        device=model.get_xyz.device)
    # We do not need to compute grad w.r.t. screenspace points
    # try:
    #     means2D.retain_grad()
    # except:
    #     pass
    means3D = model.get_xyz
    opacities = model.get_opacity
    scales = model.get_scaling
    rotations = model.get_rotation
    radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    rets = {"viewspace_points": means2D,
            "visibility_filter": radii > 0,
            "radii": radii}
    render_alpha = allmap[1:2]
    mask = (render_alpha > 0.0).squeeze(0)

    # Get Normal map
    render_normal = allmap[2:5]
    # Transform to world space
    render_normal = (
        render_normal.permute(1, 2, 0) @
        camera.world_view_transform[:3, :3].T
    ).permute(2, 0, 1)
    render_normal[..., mask] = render_normal[..., mask] / \
        render_alpha[..., mask]

    # Get Median depth
    render_depth_median = allmap[5:6]
    # render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # Get Expected depth
    render_depth_expected = allmap[0:1]
    render_depth_expected[..., mask] = render_depth_expected[..., mask] / \
        render_alpha[..., mask]
    # In theory useless
    # render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # Get distortion map
    rend_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1 - depth_ratio) + \
        render_depth_median * depth_ratio

    surf_normal = depth_to_normal(camera, surf_depth)
    # surf_normal[..., mask] = surf_normal[..., mask] / render_alpha[..., mask]
    rets.update(
        {
            "rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": rend_dist,
            "surf_depth": surf_depth,
            "surf_normal": surf_normal
        }
    )
    return rets
