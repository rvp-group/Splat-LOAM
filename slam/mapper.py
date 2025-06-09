import torch
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
        self.prune()
        model_size_mb = self.model.size_mb
        model_no_gaussians = self.model.no_gaussians
        logger.info(f"Model updated. | No. primitives = {model_no_gaussians}, "
                    f"{model_size_mb:.2f} MB")
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
        if no_samples < 2:
            return

        depth_gradient = compute_depth_gradient(
            frame.camera.image_depth, frame.camera.image_valid)
        depth_gradient = depth_gradient / depth_gradient.max()
        # Typically happens when there's nothing to densify
        if depth_gradient[..., densify_mask].sum() <= 1e-5:
            return
        sampled_indices = torch.multinomial(
            depth_gradient[..., densify_mask],
            no_samples
        )
        densify_mask_sampled = torch.zeros_like(densify_mask)
        densify_mask_sampled[candidates[sampled_indices, 0],
                             candidates[sampled_indices, 1]] = 1.0

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
        normals = frame.model_T_frame[:3, :3] @ normals
        n_R = create_rotation_matrix_from_direction_vector_batch(normals.T)
        rots = matrix_to_quaternion(n_R)
        opacities = self.model.get_gmodel.opacity_inverse_activation(
            0.9 * torch.ones((num_newpoints, 1),
                             dtype=torch.float32,
                             device=self.cfg.device)
        )
        logger.info(f"Adding "
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
            self.model.get_gmodel.optimizer.zero_grad(set_to_none=True)
            keyframe = samplers.sample_uniform(self.model.keyframes)
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
                est_depth - gt_depth)).mean()
            # Eq (15)
            normal_loss = (1 - (
                est_normal[..., valid_mask] * surf_normal[..., valid_mask]
            ).sum(dim=0)).mean()
            normal_loss *= self.cfg.mapping.opt_lambda_normal
            # Eq (16)
            alpha_loss = torch.nn.functional.binary_cross_entropy(
                est_alpha[..., valid_mask],
                gt_alpha[..., valid_mask].float(),
                reduction="mean"
            )
            alpha_loss *= self.cfg.mapping.opt_lambda_alpha

            # Eq (17)
            scales_max = self.model.get_gmodel.get_scaling.max(dim=1).values
            opt_scaling_max = self.cfg.mapping.opt_scaling_max
            reg_scales = scales_max[scales_max >=
                                    opt_scaling_max] - opt_scaling_max
            reg_scales = (self.cfg.mapping.opt_scaling_max_penalty *
                          reg_scales).sum()

            loss_total = geom_l1 + \
                alpha_loss + \
                normal_loss + \
                reg_scales
            loss_total.backward()

            with torch.no_grad():
                self.model.get_gmodel.optimizer.step()
                if loss_ema is None:
                    loss_ema = loss_total.item()
                else:
                    loss_ema = (
                        loss_ema_alpha * loss_total.item() +
                        (1 - loss_ema_alpha) * loss_ema
                    )
                if (iter + 1) % 100 == 0:
                    logger.debug(f"it={iter+1} l_ema={loss_ema:.3f}")
                # if (iter + 1) % 10 == 0:
                #     logger.debug(f"(it={iter+1}) loss={loss_ema:.4f}")
                #     rr.log("rend_depth", rr.DepthImage(
                #         est_depth.cpu().numpy()
                #     ))
                #     rr.log("rend_normal", rr.Image(
                #         ((est_normal.permute(1, 2, 0).cpu().numpy()
                #           * 0.5 + 0.5) * 255).astype(np.uint8)
                #     ))
                #     rr.log("surf_normal", rr.Image(
                #         ((surf_normal.permute(1, 2, 0).cpu().numpy()
                #           * 0.5 + 0.5) * 255).astype(np.uint8)
                #     ))
                #     rr.log("valid_mask", rr.Image(
                #         valid_mask.cpu().numpy().astype(np.uint8) * 255
                #     ))
                #     rr.log("est_alpha", rr.Image(
                #         (est_alpha[0].cpu().numpy() * 255).astype(np.uint8)
                #     ))
                #     depth_l1 = torch.abs(gt_depth - est_depth)
                #     depth_l1[..., ~valid_mask] = 0.0
                #     rr.log("depth_l1", rr.DepthImage(
                #         depth_l1.cpu().numpy()
                #     ))
                #     # Log model
                #     gmodel = self.model.get_gmodel
                #     centers = gmodel.get_xyz.cpu().numpy()
                #     hsize = gmodel.get_scaling.cpu().numpy()
                #     hsize = np.concatenate(
                #         [3.3 * hsize, 0.001 * np.ones((hsize.shape[0], 1))], axis=1)
                #     q = gmodel.get_rotation.cpu().numpy()
                #     q = np.concatenate([q[:, 1:], q[:, 0:1]], axis=1)
                #     opacity = gmodel.get_opacity.cpu().numpy().squeeze()
                #     rr.log("model",
                #            rr.Ellipsoids3D(
                #                centers=centers,
                #                half_sizes=hsize,
                #                quaternions=q,
                #                fill_mode=rr.components.FillMode.Solid,
                #            ))
        return

    def prune(self):
        """
        Prune gaussians based on several factors:
        1) opacity < cfg.mapping.pruning_min_opacity (if > 0)
        2) norm(scaling) < cfg.mapping.pruning_min_size (if > 0)
        ...
        """
        opacity = self.model.get_gmodel.get_opacity
        min_opacity = self.cfg.mapping.pruning_min_opacity
        prune_mask = torch.zeros_like(opacity, dtype=torch.bool)
        if min_opacity > 0:
            prune_mask = prune_mask | (opacity < min_opacity)
        scaling = self.model.get_gmodel.get_scaling
        min_scaling = self.cfg.mapping.pruning_min_size
        if min_scaling > 0:
            prune_mask = prune_mask | (torch.linalg.norm(
                scaling, dim=-1) < min_scaling)
        logger.info(f"Pruning {prune_mask.sum()} gaussians")
        self.model.get_gmodel.prune_points(prune_mask.squeeze())
