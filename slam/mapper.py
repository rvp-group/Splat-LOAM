import torch
import numpy as np
from utils.config_utils import Configuration
from utils.logging_utils import get_logger
from utils.graphic_utils import depth_to_points, compute_depth_gradient
from utils.general_utils import (
    create_rotation_matrix_from_direction_vector_batch,
    matrix_to_quaternion
)
from slam.local_model import LocalModel
from scene.frame import Frame
from gaussian_renderer import render
from simple_knn._C import distCUDA2
import utils.sampling_utils as samplers
import rerun as rr

logger = get_logger("mapper")


class Mapper:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.model: LocalModel = None
        ...

    def register_model(self, model: LocalModel) -> None:
        """
        Change internal reference to the active local model
        """
        self.model = model
        return

    def update_model(self, frame: Frame,
                     initialize_model: bool = False) -> None:
        """
        Perform model update via:
        - Densification
        - Gaussian update via model's keyframes
        """
        self.densify(frame, initialize_model)
        self.optimize()
        return

    @torch.no_grad()
    def densify(self, frame: Frame, initialize_model: bool = False) -> None:
        if initialize_model is False:
            render_pkg = render(frame.camera,
                                self.model.get_gmodel,
                                self.cfg.opt.depth_ratio)
            # Identify pixels whose opacity is low but
            # are available in the measurement
            mask_opacity = (
                render_pkg["rend_alpha"][0] <=
                self.cfg.mapping.densify_threshold_opacity
            )
            valid_pixels = frame.camera.image_valid[0] == 1.0
            densify_mask = (mask_opacity & valid_pixels)

            if self.cfg.mapping.densify_threshold_egeom > 0.0:
                # Identify pixels for which estimated depths are larger
                # than gt depths
                # and where depth loss is larger than a quantile
                est_depth = render_pkg["surf_depth"]
                gt_depth = frame.camera.image_depth
                geom_loss = torch.abs(gt_depth - est_depth)
                geom_loss[..., ~valid_pixels] = 0.0
                mask_depth = (est_depth > gt_depth) * (
                    geom_loss > geom_loss.quantile(0.95)
                )
                densify_mask = densify_mask | mask_depth[0]
        else:
            densify_mask = frame.camera.image_valid[0]

        candidates = densify_mask.nonzero()
        no_samples = int(self.cfg.mapping.densify_percentage *
                         candidates.shape[0])
        # No densification can occurr here
        if no_samples == 0:
            return

        depth_gradient = compute_depth_gradient(
            frame.camera.image_depth, frame.camera.image_valid)
        depth_gradient = depth_gradient / depth_gradient.max()
        sampled_indices = torch.multinomial(
            depth_gradient[..., densify_mask],
            no_samples
        )
        densify_mask_sampled = torch.zeros_like(densify_mask)
        densify_mask_sampled[candidates[sampled_indices, 0],
                             candidates[sampled_indices, 1]] = 1.0

        rr.log("densification_mask", rr.Image(
            densify_mask_sampled.cpu().numpy().astype(np.uint8) * 255
        ))

        # Generate point cloud for densifications
        points = depth_to_points(frame.camera,
                                 frame.camera.image_depth)
        points = points[..., densify_mask_sampled].T
        num_newpoints = points.shape[0]
        if initialize_model:
            full_xyz = points
        else:
            full_xyz = torch.cat((
                points, self.model.get_gmodel.get_xyz
            ))

        dist2 = torch.clamp_max(
            torch.clamp_min(distCUDA2(full_xyz), 1e-7),
            self.cfg.mapping.opt_scaling_max**2)[:num_newpoints]

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        # Align splats to normals
        # The normals are expressed in sensor frame, therefore we express them
        # in world frame
        normals = frame.camera.image_normal[..., densify_mask_sampled]
        normals = frame.model_T_cam[:3, :3] @ normals
        n_R = create_rotation_matrix_from_direction_vector_batch(normals.T)
        rots = matrix_to_quaternion(n_R)
        opacities = self.model.get_gmodel.opacity_inverse_activation(
            0.9 * torch.ones((num_newpoints, 1),
                             dtype=torch.float32,
                             device=self.cfg.device)
        )
        logger.info(f"Densifiyng model with "
                    f"{num_newpoints} new gaussians")
        self.model.get_gmodel.densification_postfix(
            new_xyz=points,
            new_opacity=opacities,
            new_scaling=scales,
            new_rotation=rots,
        )
        return

    def optimize(self) -> None:
        loss_ema = None
        loss_ema_alpha = 0.1
        for iter in range(self.cfg.mapping.num_iterations + 1):
            # keyframe = samplers.sample_uniform(self.model.keyframes)
            keyframe = self.model.keyframes[-1]
            render_pkg = render(keyframe.camera,
                                self.model.get_gmodel,
                                self.cfg.opt.depth_ratio)
            # Compute Loss
            (est_alpha, est_depth, est_normal, surf_normal) = (
                render_pkg["rend_alpha"],
                render_pkg["surf_depth"],
                render_pkg["rend_normal"],
                render_pkg["surf_normal"]
            )
            gt_alpha, gt_depth = keyframe.camera.image_valid, \
                keyframe.camera.image_depth
            valid_mask = gt_alpha[0] == 1.0

            geom_l1 = torch.abs(valid_mask * (
                gt_depth - est_depth)).mean()
            # Eq (15)
            normal_loss = (1 - (
                est_normal[..., valid_mask] * surf_normal[..., valid_mask]
            ).sum(dim=0)).mean()
            # Eq (16)
            alpha_loss = torch.nn.functional.binary_cross_entropy(
                est_alpha[..., valid_mask],
                gt_alpha[..., valid_mask].float(),
                reduction="mean"
            )

            # Eq (17)
            scales, _ = self.model.get_gmodel.get_scaling.max(dim=1)
            scaling_max = self.cfg.mapping.opt_scaling_max
            reg_scaling = scales[scales >= scaling_max] - scaling_max
            reg_scaling = reg_scaling.sum() * \
                self.cfg.mapping.opt_scaling_max_penalty

            loss_total = geom_l1 + \
                self.cfg.mapping.opt_lambda_alpha * alpha_loss + \
                self.cfg.mapping.opt_lambda_normal * normal_loss + \
                reg_scaling
            loss_total.backward()

            with torch.no_grad():
                if loss_ema is None:
                    loss_ema = loss_total.item()
                else:
                    loss_ema = (
                        loss_ema_alpha * loss_total.item() +
                        (1 - loss_ema_alpha) * loss_ema
                    )
                if (iter + 1) % 100 == 0:
                    logger.debug(f"[it={iter+1}] loss={loss_total:.4f} "
                                 f"geom={geom_l1:.4f} "
                                 f"normal={normal_loss:.4f} "
                                 f"alpha={alpha_loss:.4f} "
                                 f"scaling={reg_scaling}")
                    self.model.get_gmodel.optimizer.step()
                    rr.log("rend_depth", rr.DepthImage(
                        est_depth.cpu().numpy()
                    ))
                    rr.log("rend_normal", rr.Image(
                        ((est_normal.permute(1, 2, 0).cpu().numpy()
                          * 0.5 + 0.5) * 255).astype(np.uint8)
                    ))
                    rr.log("surf_normal", rr.Image(
                        ((surf_normal.permute(1, 2, 0).cpu().numpy()
                          * 0.5 + 0.5) * 255).astype(np.uint8)
                    ))
                    rr.log("valid_mask", rr.Image(
                        valid_mask.cpu().numpy().astype(np.uint8) * 255
                    ))
                    rr.log("est_alpha", rr.Image(
                        (est_alpha[0].cpu().numpy() * 255).astype(np.uint8)
                    ))

        return
