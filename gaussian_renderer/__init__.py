import torch
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.config_utils import PipelineConfig
from typing import Optional


def render(
        camera: Camera,
        model: GaussianModel,
        pipe: Optional[PipelineConfig] = None):

    raster_settings = ...

    rasterizer = ...

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
    # TODO: Continue implementing this
    raise NotImplementedError("Still TODO")

    ...
