import torch
from typing import Protocol


class DataLoggerProtocol(Protocol):

    def set_timestamp(self, timestamp: float):
        ...

    def log_image(self, name: str, image: torch.Tensor) -> None:
        ...

    def log_depth_image(self, name: str, image: torch.Tensor) -> None:
        ...

    def log_model(self, name: str, value: ...) -> None:
        ...

    def log_transform(self, name: str, pose: torch.Tensor) -> None:
        ...

    def log_pointcloud(self, name: str, cloud: torch.Tensor) -> None:
        ...
