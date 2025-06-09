import torch
from utils.config_utils import Configuration
from scene.frame import Frame
from slam.mapper import Mapper
from slam.tracker import Tracker
from slam.local_model import LocalModel
from utils.logging_utils import get_logger

logger = get_logger("slam")


class SLAM:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.mapper = Mapper(cfg)
        self.tracker = Tracker(cfg)
        self.local_models: list[LocalModel] = []
        self.frames = []

    def process(self, frame: Frame) -> None:
        # Initialize the system.
        if len(self.frames) == 0:
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
            self.insert_new_keyframe(frame)

        self.frames.append(frame)

        torch.cuda.empty_cache()

    def insert_new_keyframe(self, frame: Frame):
        logger.info("Inserting new keyframe")
        self.local_models[-1].insert_keyframe(frame)
        self.mapper.update_model(frame)
        self.tracker.register_keyframe(frame)

    def initialize_new_local_model(self, frame: Frame):
        logger.info("Inserting new local model")
        lmodel = LocalModel(self.cfg)
        lmodel.insert_keyframe(frame)
        self.mapper.register_model(lmodel)
        self.mapper.update_model(frame, initialize_model=True)
        self.tracker.register_model(lmodel)
        self.tracker.register_keyframe(frame)
        self.local_models.append(lmodel)
        self.frames.append(frame)
