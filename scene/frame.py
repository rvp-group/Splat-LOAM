import torch
import numpy as np
from scene.cameras import Camera


class Frame:
    def __init__(self, camera: Camera, timestamp: float, device: str,
                 model_T_frame: np.ndarray | None = None,
                 world_T_frame: np.ndarray | None = None):
        """
        In a nutshel:
        model_T_frame should be estimated by SplatLOAM
        world_T_frame should be provided via ground-truth
        """
        self.camera = camera
        self.timestamp = timestamp
        model_T_frame = np.eye(4) if model_T_frame is None else model_T_frame
        world_T_frame = np.eye(4) if world_T_frame is None else world_T_frame
        self.model_T_frame = torch.from_numpy(model_T_frame).float().to(device)
        self.world_T_frame = torch.from_numpy(world_T_frame).float().to(device)

    def streamOut(self):
        """
        Stream out memory from GPU
        """
        self.camera.streamOut()
        self.model_T_frame = self.model_T_frame.cpu()
        torch.cuda.empty_cache()
