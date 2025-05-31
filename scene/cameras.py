import torch
from torch import nn
import numpy as np
from utils.logging_utils import get_logger
from utils.graphic_utils import getWorld2View2

logger = get_logger("")


class Camera(nn.Module):
    def __init__(self, K: np.ndarray,
                 image_depth: np.ndarray,
                 image_normal: np.ndarray,
                 image_valid: np.ndarray,
                 world_T_lidar: np.ndarray = np.eye(4, dtype=np.float32),
                 data_device: str = "cuda"):
        """
        This class models a single LiDAR measurement

        Args:
            K: (3, 3) float32 camera matrix
            image_depth: (1, H, W) float32 range image
            image_normal: (3, H, W) float32 normal image
            image_valid: (1, H, W) uint8 valid image
            world_T_lidar: (4, 4) float32 transformation matrix from LiDAR to
                World frame.
        """

        super(Camera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            logger.warning(f"{e} | falling back to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_depth = torch.from_numpy(image_depth).to(self.data_device)
        self.image_normal = torch.from_numpy(image_normal).to(self.data_device)
        self.image_valid = torch.from_numpy(image_valid).to(self.data_device)

        self.image_width = self.image_depth.shape[2]
        self.image_height = self.image_depth.shape[1]
        self.world_view_transform = torch.tensor(
            getWorld2View2(world_T_lidar[:3, :3],
                           world_T_lidar[:3, -1]))\
            .transpose(0, 1).to(data_device)
        K = torch.from_numpy(K).float().to(data_device)
        self.projection_matrix = torch.eye(4, dtype=torch.float32,
                                           device=data_device)
        self.projection_matrix[:3, :3] = K.transpose(0, 1)
