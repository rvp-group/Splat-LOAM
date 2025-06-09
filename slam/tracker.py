import torch
from typing import Protocol
from utils.config_utils import Configuration, TrackingMethod
from gsaligner import GSAligner, GSAlignerParams
from scene.frame import Frame
from gaussian_renderer import render
from slam.local_model import LocalModel
from utils.logging_utils import get_logger
from utils.graphic_utils import depth_to_points, getWorld2View2
import rerun as rr
logger = get_logger("tracker")


class Tracker:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.model: LocalModel = None
        self.num_frames_tracked = 0
        self.keyframe_T_frame = torch.eye(4, dtype=torch.float32,
                                          device=cfg.device)
        self.aligner: Aligner = aligner_available[cfg.tracking.method](cfg)

    def register_model(self, model: LocalModel) -> None:
        """
        Change internal reference to the active local model
        """
        self.model = model
        self.num_frames_tracked = 0
        self.aligner.set_model(model)
        return

    @torch.no_grad()
    def register_keyframe(self, keyframe: Frame) -> None:
        """
        Should be called when a new keyframe is set.
        Initializes a new local anchor for the tracker.
        """
        self.keyframe_T_frame.zero_()
        self.keyframe_T_frame.fill_diagonal_(1.0)
        self.aligner.set_target(keyframe)
        self.num_frames_tracked = 0

    @torch.no_grad()
    def track(self, frame: Frame) -> None:
        rr.set_time("tracking", timestamp=frame.timestamp)
        self.aligner.set_source(frame)
        self.keyframe_T_frame = self.aligner.align(
            self.keyframe_T_frame).to(self.cfg.device)

        model_T_keyframe = self.model.keyframes[-1].model_T_frame
        model_T_frame = model_T_keyframe @ self.keyframe_T_frame
        # Update camera pose
        frame.camera.world_view_transform = torch.linalg.inv(
            model_T_frame).transpose(0, 1)
        frame.model_T_frame = model_T_frame
        self.num_frames_tracked += 1
        logger.info("Tracked new frame | "
                    f"position={model_T_frame[:3, -1].cpu().numpy()} "
                    f"fitness={self.aligner.fitness():.3f}")
        rr.log("world/frame", rr.Transform3D(
            translation=model_T_frame[:3, -1].cpu().numpy(),
            mat3x3=model_T_frame[:3, :3].cpu().numpy(),
            axis_length=0.5
        ))
        return

    def require_new_keyframe(self):
        """
        Returns whether a new keyframe may be required based on several
        conditions:
        1) no_frames_tracked > threshold_nframes (if >0)
        2) registration fitness < threshold_fitness (if >0)
        3) keyframe - frame distance > threshold_distance (if >0)
        """
        threshold_nframes = self.cfg.tracking.keyframe_threshold_nframes
        threshold_fitness = self.cfg.tracking.keyframe_threshold_fitness
        threshold_distance = self.cfg.tracking.keyframe_threshold_distance
        no_frames = self.num_frames_tracked
        fitness = self.aligner.fitness()
        kf_f_distance = torch.linalg.norm(self.keyframe_T_frame[:3, -1])
        return ((threshold_nframes > 0) and
                (no_frames > threshold_nframes)) or \
            ((threshold_fitness > 0) and (fitness < threshold_fitness)) or \
            ((threshold_distance > 0) and (kf_f_distance > threshold_distance))


class Aligner(Protocol):
    def set_source(self, frame: Frame) -> None:
        """ Setup source frame (typically the new frame) """

    def set_target(self, frame: Frame) -> None:
        """ Setup the target frame (typically the fixed keyframe) """

    def align(self, iguess: torch.Tensor) -> torch.Tensor:
        """ Estimates target_T_source transform matrix """

    def fitness(self) -> float:
        """ Returns the fitness of the last alignment """

    def set_model(self, model: LocalModel) -> None:
        """ Update internal representation of local model """


class AlignerGT:
    def __init__(self, cfg: Configuration):
        self.source_camera = None
        self.target_camera = None

    def set_source(self, frame: Frame) -> None:
        """ Setup source frame (typically the new frame) """
        self.source_camera = frame.camera
        depth = frame.camera.image_depth
        points = depth_to_points(frame.camera,
                                 depth,
                                 transform_in_world=False)
        points = points.permute(1, 2, 0).reshape(-1, 3)
        rr.log("world/frame/cloud", rr.Points3D(
            positions=points.cpu().numpy()))

    def set_target(self, frame: Frame) -> None:
        """ Setup the target frame (typically the fixed keyframe) """
        self.target_camera = frame.camera

    def align(self, iguess: torch.Tensor) -> torch.Tensor:
        """ Estimates target_T_source transform matrix """
        assert self.source_camera and self.target_camera
        target_T_world = self.target_camera.world_view_transform
        target_T_world = target_T_world.transpose(0, 1)
        world_T_source = self.source_camera.world_view_transform
        world_T_source = torch.linalg.inv(world_T_source).transpose(0, 1)
        target_T_source = target_T_world @ world_T_source
        return target_T_source

    def fitness(self) -> float:
        """ Returns the fitness of the last alignment """
        return 1.0

    def set_model(self, model: LocalModel) -> None:
        """ Update internal representation of local model """
        self.model = model
        return


class AlignerGeomPhoto:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.reg_fitness = 1.0
        self.model: LocalModel = None
        gsaligner_params = GSAlignerParams()
        if cfg.tracking.gsaligner:
            gsaligner_params = cfg.tracking.gsaligner
        else:
            logger.warning(
                "configuration lacks tracking.gsaligner [GSAlignerParams]. "
                "Using default parameters.")
        # Enforce image sizes
        gsaligner_params.image_height = cfg.preprocessing.image_height
        gsaligner_params.image_width = cfg.preprocessing.image_width
        logger.debug(f"Initializing gsaligner with parameters: "
                     f"{gsaligner_params.__dict__}")
        self.gsaligner = GSAligner(**gsaligner_params.__dict__)

    def set_source(self, frame: Frame) -> None:
        """ Setup source frame (typically the new frame) """
        assert self.model is not None
        depth = frame.camera.image_depth
        points = depth_to_points(frame.camera,
                                 depth,
                                 transform_in_world=False)
        points = points.permute(1, 2, 0).reshape(-1, 3)
        rr.log("world/frame/cloud", rr.Points3D(
            positions=points.cpu().numpy()))
        self.gsaligner.set_query(depth, points, frame.camera.projection_matrix)

    def set_target(self, frame: Frame) -> None:
        """ Setup the target frame (typically the fixed keyframe) """
        assert self.model is not None
        render_pkg = render(frame.camera,
                            self.model.get_gmodel,
                            self.cfg.opt.depth_ratio)
        depth = render_pkg["surf_depth"]
        points = depth_to_points(frame.camera, depth,
                                 transform_in_world=False)
        points = points.permute(1, 2, 0).reshape(-1, 3)
        self.gsaligner.set_reference(
            depth, points, frame.camera.projection_matrix)

    def align(self, iguess: torch.Tensor) -> torch.Tensor:
        """ Estimates target_T_source transform matrix """
        assert self.model is not None
        keyframe_T_frame, fitness, _ = self.gsaligner.align(iguess)
        self.reg_fitness = fitness
        return keyframe_T_frame

    def fitness(self) -> float:
        """ Returns the fitness of the last alignment """
        assert self.model is not None
        return self.reg_fitness

    def set_model(self, model: LocalModel) -> None:
        """ Update internal representation of local model """
        self.model = model


aligner_available = {
    TrackingMethod.gsaligner: AlignerGeomPhoto,
    TrackingMethod.gt: AlignerGT
}
