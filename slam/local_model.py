import torch
from utils.config_utils import Configuration
from scene.gaussian_model import GaussianModel
from scene.frame import Frame


class LocalModel:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.keyframes = []
        self.world_T_model = torch.eye(4)
        self.model = GaussianModel(self.cfg.device)
        self.model.training_setup(cfg)

    def insert_keyframe(self, frame: Frame) -> None:
        self.keyframes.append(frame)
        return

    def require_new_model(self) -> bool:
        """
        Returns true if local model is full or
        if any conditions are met for new model spawn
        """
        ...
        return False

    @property
    def get_gmodel(self):
        return self.model
