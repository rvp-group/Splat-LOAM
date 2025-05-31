import torch
from scene.cameras import Camera


class Frame:
    def __init__(self, camera: Camera, timestamp: float, device: str):
        self.camera = camera
        self.timestamp = timestamp
        self.model_T_cam = torch.eye(4, dtype=torch.float32,
                                     device=device)

    def streamOut(self):
        """
        Stream out memory from GPU
        """
        self.camera.streamOut()
        self.model_T_cam = self.model_T_cam.cpu()
        torch.cuda.empty_cache()
