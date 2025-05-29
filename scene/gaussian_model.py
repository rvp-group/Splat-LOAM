import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    matrix_to_quaternion,
    create_rotation_matrix_from_direction_vector_batch
)
from utils.graphic_utils import BasicPointCloud
from simple_knn._C import distCUDA2
from utils.config_utils import Configuration


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(
            center: torch.Tensor,
            scaling: torch.Tensor,
            rotation: torch.Tensor,
        ):
            RS = build_scaling_rotation(
                torch.cat(
                    [scaling, torch.ones_like(scaling)], dim=-1
                ),
                rotation,
            ).permute(0, 2, 1)
            trans = torch.zeros(
                (center.shape[0], 4, 4), dtype=torch.float, device=self.device)
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            return trans
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.optimizer = None
        self.setup_functions()

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier: float = 1) -> torch.Tensor:
        return self.covariance_activation(self._scaling, scaling_modifier)

    def create_from_pcd(self, pcd: BasicPointCloud) -> None:
        cloud_xyz = torch.tensor(np.asarray(
            pcd.points)).float().to(self.device)
        dist2 = torch.clamp_max(
            torch.clamp_min(
                distCUDA2(cloud_xyz),
                1e-7
            ), 0.5)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        # Align splats to normals
        n_R = create_rotation_matrix_from_direction_vector_batch(
            torch.from_numpy(pcd.normals).float().to(self.device)
        )
        rots = matrix_to_quaternion(n_R)
        opacities = self.opacity_inverse_activation(
            0.9 *
            torch.ones((cloud_xyz.shape[0], 1),
                       dtype=torch.float32, device=self.device))
        self._xyz = nn.Parameter(cloud_xyz.requires_grad(True))
        self._scaling = nn.Parameter(scales.requires_grad(True))
        self._rotation = nn.Parameter(rots.requires_grad(True))
        self._opacity = nn.Parameter(opacities.requires_grad(True))

    def training_setup(self, cfg: Configuration):
        optim_cfg = cfg.opt
        training_vars = [
            {
                "params": [self._xyz],
                "lr": optim_cfg.position_lr,
                "name": "xyz",
            },
            {
                "params": [self._opacity],
                "lr": optim_cfg.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": optim_cfg.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": optim_cfg.rotation_lr,
                "name": "rotation",
            }
        ]
        self.optimizer = torch.optim.Adam(training_vars, lr=0.0, eps=1e-15)
