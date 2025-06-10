import torch
from utils.config_utils import Configuration
from scene.gaussian_model import GaussianModel
from scene.frame import Frame


class LocalModel:
    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.keyframes: list[Frame] = []
        self.world_T_model = torch.eye(4)
        self.model = GaussianModel(self.cfg.device)
        self.model.training_setup(cfg)

    def insert_keyframe(self, frame: Frame) -> None:
        self.keyframes.append(frame)
        return

    def require_new_model(self) -> bool:
        """
        Returns whether a new model is required based on several
        conditions:
        1) no_gaussians > mapping.lmodel_threshold_ngaussians (if >0)
        2) no_keyframes > mapping.lmodel_threshold_nkeyframes (if >0)
        """
        threshold_ngaussians = self.cfg.mapping.lmodel_threshold_ngaussians
        threshold_nkeyframes = self.cfg.mapping.lmodel_threshold_nkeyframes
        ret = False

        if threshold_ngaussians and threshold_ngaussians > 0:
            ret = ret or (self.no_gaussians > threshold_ngaussians)
        if threshold_nkeyframes and threshold_nkeyframes > 0:
            ret = ret or (len(self.keyframes) > threshold_nkeyframes)

        return ret

    @property
    def get_gmodel(self):
        return self.model

    @property
    def size_mb(self):
        no_fields = 3 * 4 * 2 * 1  # xyz + rots + scales + opacity
        field_size = 4  # bytes per float32 element
        return (no_fields * field_size * self.no_gaussians) / (1024.**2)

    @property
    def no_gaussians(self):
        return self.model.get_xyz.shape[0]
