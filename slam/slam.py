import torch
import numpy as np
from utils.trajectory_utils import (
    TrajectoryWriterType,
    trajectory_writer_available)
from datetime import datetime
from pathlib import Path
from utils.config_utils import Configuration, save_configuration
from scene.frame import Frame
from slam.mapper import Mapper
from slam.tracker import Tracker
from slam.local_model import LocalModel
from utils.graphic_utils import depth_to_points
from utils.logging_utils import get_logger
from utils.logging_backends import get_datalogger
from gaussian_renderer import render
from scene.postprocessing import ResultGraph

logger = get_logger("slam")


class SLAM:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.mapper = Mapper(cfg)
        self.tracker = Tracker(cfg)
        self.local_models: list[LocalModel] = []
        self.frames: list[Frame] = []
        # Required to output results
        self.date_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.world_T_odom: list[np.array] = []

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
            self.world_T_odom.append(wTf.cpu().numpy())
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

    def save_results(self) -> Path:
        """
        Save output of SLAM system.
        Specifically, it stores:
        - configuration (cfg.yaml)
        - odometry results (odom.txt) [uses output.writer for format]
        - models-keyframes graph (graph.yaml) [refer to scene.postprocessing]
        - models/ : local models as PLY files
        """
        ofolder = self.cfg.output.folder
        result_folder = "results/" if ofolder is None else \
            ofolder

        result_folder = Path(result_folder)
        result_folder = result_folder / self.date_start
        result_folder.mkdir(parents=True, exist_ok=False)
        logger.info(f"Saving results in {result_folder}")
        result_models_folder = result_folder / "models"
        result_models_folder.mkdir(parents=True, exist_ok=True)
        save_configuration(result_folder / "cfg.yaml", self.cfg)
        writer_type = self.cfg.output.writer
        if writer_type is None:
            logger.debug("output.writer is not set. Assuming tum")
            writer_type = TrajectoryWriterType.tum

        trajectory_writer = trajectory_writer_available[writer_type]
        timestamps = [f.timestamp for f in self.frames]
        trajectory_writer.write(result_folder / "odom.txt", self.world_T_odom,
                                timestamps)

        rgraph = ResultGraph.from_slam(self.cfg,
                                       self.local_models,
                                       Path("models"))
        save_configuration(result_folder / "graph.yaml", rgraph)
        for i, rmodel in enumerate(rgraph.models):
            gmodel = self.local_models[i].get_gmodel
            filename = rmodel.filename
            gmodel_path = result_folder / filename
            logger.debug(f"Saving model at {gmodel_path}")
            gmodel.save_ply(gmodel_path)
        return result_folder
