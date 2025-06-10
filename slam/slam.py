import torch
from utils.config_utils import Configuration
from scene.frame import Frame
from slam.mapper import Mapper
from slam.tracker import Tracker
from slam.local_model import LocalModel
from utils.graphic_utils import depth_to_points
from utils.logging_utils import get_logger
from utils.logging_backends import get_datalogger
from gaussian_renderer import render

logger = get_logger("slam")


class SLAM:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.mapper = Mapper(cfg)
        self.tracker = Tracker(cfg)
        self.local_models: list[LocalModel] = []
        self.frames = []

    def process(self, frame: Frame) -> None:
        get_datalogger(self.cfg).set_timestamp(frame.timestamp)
        # Initialize the system.
        if len(self.frames) == 0:
            if frame.world_T_frame is not None:
                # Align to GT if first frame
                frame.model_T_frame = frame.world_T_frame.clone()
            self.initialize_new_local_model(frame)
            return

        # Tracking
        self.tracker.track(frame)

        # Update keyframe if needed
        if self.tracker.require_new_keyframe():
            logger.debug("New keyframe required")
            # Handle new keyframe
            if self.local_models[-1].require_new_model():
                # Update local model before optimization
                self.initialize_new_local_model(frame)
            else:
                self.insert_new_keyframe(frame)

        self.frames.append(frame)

        # Log transforms, models and images
        with torch.no_grad():
            wTm = self.local_models[-1].world_T_model
            mTkf = self.local_models[-1].keyframes[-1].model_T_frame
            kfTf = self.tracker.keyframe_T_frame
            wTf = wTm @ mTkf @ kfTf
            logger.info(f"t={frame.timestamp} | "
                        f"pos={wTf[:3, -1].cpu().numpy()}")
            dlog = get_datalogger(self.cfg)
            dlog.log_transform("world/model", wTm)
            dlog.log_transform("world/model/keyframe", mTkf)
            dlog.log_transform("world/model/keyframe/frame", kfTf)
            points_in = depth_to_points(frame.camera,
                                        frame.camera.image_depth,
                                        transform_in_world=False)
            dlog.log_pointcloud(
                "world/model/keyframe/frame", points_in.reshape(3, -1).T)
            render_pkg = render(frame.camera, self.local_models[-1].get_gmodel,
                                self.cfg.opt.depth_ratio)
            gt_depth, est_depth = frame.camera.image_depth, \
                render_pkg["surf_depth"]
            depth_l1 = torch.abs(est_depth - gt_depth)
            depth_l1[frame.camera.image_valid == 0] = 0.0
            est_normal = render_pkg["rend_normal"]*0.5 + 0.5
            dlog.log_image("frame/normals", est_normal)
            dlog.log_depth_image("frame/depth_in", gt_depth)
            dlog.log_depth_image("frame/depth", est_depth)
            dlog.log_depth_image("frame/depth_l1", depth_l1)

        torch.cuda.empty_cache()

    def insert_new_keyframe(self, frame: Frame):
        logger.info("Inserting new keyframe")
        self.local_models[-1].insert_keyframe(frame)
        self.mapper.update_model(frame)
        self.tracker.register_keyframe(frame)
        with torch.no_grad():
            get_datalogger(self.cfg).log_model(
                "world/model", self.local_models[-1].get_gmodel)

    def initialize_new_local_model(self, frame: Frame):
        logger.info("Inserting new local model")
        lmodel = LocalModel(self.cfg)
        # Override the local model origin and
        # its first keyframe origin
        if len(self.local_models) == 0:
            world_T_lmodel_old = torch.eye(4).float().to(self.cfg.device)
        else:
            world_T_lmodel_old = self.local_models[-1].world_T_model
        lmodel.world_T_model = world_T_lmodel_old @ \
            frame.model_T_frame.clone()
        frame.model_T_frame.zero_()
        frame.model_T_frame.fill_diagonal_(1.0)
        frame.camera.world_view_transform.zero_()
        frame.camera.world_view_transform.fill_diagonal_(1.0)
        lmodel.insert_keyframe(frame)
        self.local_models.append(lmodel)
        self.mapper.register_model(lmodel)
        self.mapper.update_model(frame, initialize_model=True)
        self.tracker.register_model(lmodel)
        self.tracker.register_keyframe(frame)
        self.frames.append(frame)
        with torch.no_grad():
            get_datalogger(self.cfg).log_model(
                "world/model", self.local_models[-1].get_gmodel)
