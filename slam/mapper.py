import torch
from utils.config_utils import Configuration
from slam.local_model import LocalModel
from scene.frame import Frame


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
        ...
        return

    def update_model(self, frame: Frame) -> None:
        """
        Perform model update via:
        - Densification
        - Gaussian update via model's keyframes
        """
        raise RuntimeError("Not implemented yet!")
        self.densify(self, frame)
        self.optimize()
        return

    def densify(self, frame: Frame) -> None:
        ...
        return

    def optimize(self) -> None:
        ...
        return
