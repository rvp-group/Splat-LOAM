import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import (
    build_scaling_rotation,
    inverse_sigmoid,
    matrix_to_quaternion,
    create_rotation_matrix_from_direction_vector_batch
)
from utils.logging_utils import get_logger
from utils.graphic_utils import BasicPointCloud
from simple_knn._C import distCUDA2
from utils.config_utils import Configuration

logger = get_logger("gaussian_model")


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
        self._xyz = torch.empty(0, dtype=torch.float32, device=device)
        self._scaling = torch.empty(0, dtype=torch.float32, device=device)
        self._rotation = torch.empty(0, dtype=torch.float32, device=device)
        self._opacity = torch.empty(0, dtype=torch.float32, device=device)
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

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor,
                                    name: str):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(
                    group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(
                group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["name"]] = group["params"][0]
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask: torch.Tensor) -> None:
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(
                group["params"][0])
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"],
                     torch.zeros_like(extension_tensor)),
                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"],
                     torch.zeros_like(extension_tensor)),
                    dim=0)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0],
                         extension_tensor),
                        dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0],
                         extension_tensor),
                        dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self,
                              new_xyz: torch.Tensor,
                              new_opacity: torch.Tensor,
                              new_scaling: torch.Tensor,
                              new_rotation: torch.Tensor) -> None:
        d = {
            "xyz": new_xyz,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
