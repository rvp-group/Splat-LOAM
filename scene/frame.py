import torch
import numpy as np
from scene.cameras import Camera


class Frame:
    def __init__(self, camera: Camera, timestamp: float, device: str,
                 model_T_frame: np.ndarray | None = None):
        self.camera = camera
        self.timestamp = timestamp
        if model_T_frame is None:
            self.model_T_frame = torch.eye(4, dtype=torch.float32,
                                           device=device)
        else:
            self.model_T_frame = torch.from_numpy(
                model_T_frame).float().to(device)

    def streamOut(self):
        """
        Stream out memory from GPU
        """
        self.camera.streamOut()
        self.model_T_frame = self.model_T_frame.cpu()
        torch.cuda.empty_cache()
