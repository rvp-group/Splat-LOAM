import torch
from utils.config_utils import Configuration
from scene.frame import Frame
from slam.local_model import LocalModel
from utils.logging_utils import get_logger

logger = get_logger("tracker")


class Tracker:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.model: LocalModel = None
        self.temp_iter = 0
        ...

    def register_model(self, model: LocalModel) -> None:
        """
        Change internal reference to the active local model
        """
        self.model = model
        self.temp_iter = 0

        ...
        return

    def track(self, frame: Frame) -> None:
        logger.warning("tracking not yet implemented")
        self.temp_iter += 1
        frame.model_T_cam = torch.linalg.inv(
            frame.camera.world_view_transform).transpose(0, 1)
        return
        raise RuntimeError("Not implemented yet!")

    def require_new_keyframe(self):
        thr_frames = self.cfg.tracking.keyframe_threshold_nframes
        if thr_frames > 0:
            if self.temp_iter > \
                    self.cfg.tracking.keyframe_threshold_nframes:
                logger.info("Requesting new keyframe")
                self.temp_iter = 0
                return True
        return False
