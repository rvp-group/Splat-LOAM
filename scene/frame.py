import torch
from scene.cameras import Camera


class Frame:
    def _init__(self, camera: Camera, device: str):
        self.camera = camera
        self.model_T_cam = torch.eye(4, dtype=torch.float32,
                                     device=device)

    def streamOut(self):
        """
        Stream out memory from GPU
        """
        self.camera.streamOut()
        self.model_T_cam = self.model_T_cam.cpu()
        torch.cuda.empty_cache()
