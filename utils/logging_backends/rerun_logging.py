import numpy as np
import torch
import rerun as rr
import rerun.blueprint as rrb
from utils.config_utils import Configuration
from utils.general_utils import build_rotation
from utils.logging_utils import get_logger

logger = get_logger("")


class DataLoggerRR:
    def __init__(self, cfg: Configuration):
        rr.init("SplatLOAM")
        # Setup the correct blueprint
        bp = rrb.Blueprint(
            rrb.Horizontal(
                contents=[
                    rrb.Vertical(
                        contents=[
                            rrb.Spatial2DView(origin="frame/depth_in"),
                            rrb.Spatial2DView(origin="frame/depth"),
                            rrb.Spatial2DView(origin="frame/normals"),
                            rrb.Spatial2DView(origin="frame/densify_mask"),
                            rrb.Spatial2DView(origin="frame/depth_l1")
                        ]
                    ),
                    rrb.Spatial3DView(origin="world/")
                ]
            )
        )
        rr.send_blueprint(bp)
        spawn_gui = cfg.logging.rerun_spawn
        serve_grpc = cfg.logging.rerun_serve_grpc
        connect_grpc = cfg.logging.rerun_connect_grpc_url is not None

        if spawn_gui:
            logger.info(rr.spawn())
        elif serve_grpc:
            logger.info(rr.serve_grpc())
        elif connect_grpc:
            logger.info(rr.connect_grpc(
                url=cfg.logging.rerun_connect_grpc_url))

    def set_timestamp(self, timestamp: float):
        """
        Set timestamp for the next logs.
        """
        rr.set_time("time", timestamp=timestamp)

    def log_image(self, name: str, image: torch.Tensor) -> None:
        """
        Log a monochromatic or rgb image.
        The function does not perform per-channel normalization.
        Ensure the image is properly normalized between (0, 1) before logging.
        """
        if image.dim() == 2:
            # Probably (HxW) tensor, lift to (1xHxW)
            image = image.unsqueeze(0)
        image = image.permute(1, 2, 0)  # (HxWxNChnls)
        rr.log(name, rr.Image(
            (image.cpu().numpy() * 255).astype(np.uint8)))

    def log_depth_image(self, name: str, image: torch.Tensor) -> None:
        if image.dim() == 2:
            # Probably (HxW) tensor, lift to (1xHxW)
            image = image.unsqueeze(0)
        image = image.permute(1, 2, 0)
        rr.log(name, rr.DepthImage(
            image.cpu().numpy()))

    def log_model(self, name: str, gaussians) -> None:
        centers = gaussians.get_xyz.cpu().numpy()
        hsize = gaussians.get_scaling.cpu().numpy()
        hsize = np.concatenate(
            [3.3 * hsize, 0.001 * np.ones((hsize.shape[0], 1))], axis=1)
        q = gaussians.get_rotation.cpu().numpy()
        q = np.concatenate([q[:, 1:], q[:, 0:1]], axis=1)
        normals = build_rotation(
            gaussians.get_rotation).cpu().numpy()[..., :3, -1]
        normals = normals * 0.5 + 0.5
        color = np.zeros((normals.shape[0], 3), dtype=np.float32)
        color[..., :3] = normals
        rr.log(name,
               rr.Ellipsoids3D(
                   centers=centers,
                   half_sizes=hsize,
                   quaternions=q,
                   colors=color,
                   fill_mode=rr.components.FillMode.Solid,
               ))

    def log_transform(self, name: str, pose: torch.Tensor) -> None:
        pose_np = pose.cpu().numpy()
        rr.log(name, rr.Transform3D(
            translation=pose_np[:3, -1], mat3x3=pose_np[:3, :3],
            axis_length=1.0))

    def log_pointcloud(self, name: str, cloud: torch.Tensor) -> None:
        rr.log(name, rr.Points3D(
            positions=cloud.cpu().numpy()))
