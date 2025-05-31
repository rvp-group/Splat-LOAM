import torch
from utils.config_utils import Configuration
from scene.frame import Frame
from slam.mapper import Mapper
from slam.tracker import Tracker
from slam.local_model import LocalModel


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
            # Handle new keyframe
            if self.local_models[-1].require_new_model():
                # Update local model before optimization
                self.initialize_new_local_model(frame)
            self.mapper.update_model(frame)

    def initialize_new_local_model(self, frame: Frame):
        lmodel = LocalModel(self.cfg)
        lmodel.insert_keyframe(frame)
        self.mapper.register_model(lmodel)
        self.mapper.update_model(frame)
        self.tracker.register_model(lmodel)
        self.local_models.append(lmodel)
