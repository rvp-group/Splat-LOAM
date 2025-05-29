import torch
from utils.config_utils import Configuration
from scene.frame import Frame
from slam.local_model import LocalModel


class Tracker:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.model: LocalModel = None
        ...

    def register_model(self, model: LocalModel) -> None:
        """
        Change internal reference to the active local model
        """
        self.model = model
        ...
        return

    def track(self, frame: Frame, timestamp: float) -> None:
        raise RuntimeError("Not implemented yet!")
        ...
        return
