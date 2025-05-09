from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from omegaconf import OmegaConf
from pathlib import Path
from utils.logging_utils import get_logger

logger = get_logger("")


class TrackingMethod(str, Enum):
    GT = "gt"
    POINT_TO_POINT = "p2point"
    POINT_TO_PLANE = "p2plane"
    GSALIGNER = "gsaligner"


class DatasetType(str, Enum):
    custom = "custom"
    vbr = "vbr"
    kitti = "kitti"
    ncd = "ncd"
    oxspires = "oxspires"


class TrajectoryReaderType(str, Enum):
    kitti = "kitti"
    tum = "tum"
    vilens = "vilens"
    null = "null"


class TrajectoryWriterType(str, Enum):
    kitti = "kitti"
    tum = "tum"


@dataclass
class TrajectoryReaderConfig:
    # Format of trajectory file
    reader_type: Optional[TrajectoryReaderType] = None
    # File containing pose information
    filename: Optional[str] = None
    # Association tolerance for timestamp (1ms by default)
    timestamp_dtol: float = 1e-3
    # If gt_T_sensor is provided as pos-quat, set this variable
    gt_T_sensor_t_xyz_q_xyzw: Optional[tuple[float]] = field(
        default_factory=tuple)
    # If gt_T_sensor is provided via KITTI calibration file
    # set this variable
    gt_T_sensor_kitti_filename: Optional[str] = None


class PointCloudReaderType(str, Enum):
    bin = "bin"
    ply = "ply"
    pcd = "pcd"
    rosbag = "rosbag"
    null = "null"


@dataclass
class PointCloudReaderConfig:
    # Folder containing pcloud data
    cloud_folder: str = ""
    # Format of pcloud files
    cloud_format: Optional[PointCloudReaderType] = None
    # If files are indexed by timestamp, extract it from filenames
    timestamp_from_filename: Optional[bool] = False
    # If timestamp file is provided, set this variable
    timestamp_filename: Optional[str] = None
    # If using bin format with format different from kitti
    # specify this variable
    bin_format: Optional[str] = "<f4"
    # If using rosbag, specify pcloud topic
    rosbag_topic: Optional[str] = None


@dataclass
class TrackingConfig:
    num_iterations: int = 10
    method: TrackingMethod = TrackingMethod.GSALIGNER
    keyframe_threshold_distance: float = 1.0
    keyframe_threshold_nframes: int = -1
    keyframe_threshold_fitness: float = -1.0


@dataclass
class MappingConfig:
    num_iterations: int = 500
    densify_threshold_egeom: float = -1
    densify_threshold_opacity: float = 0.5
    densify_percentage: float = 0.15
    prob_view_last_keyframe: Optional[float] = 0.4
    pruning_min_opacity: float = 0.0
    pruning_min_size: Optional[float] = 0.0
    pruning_max_size: Optional[float] = 1.0
    early_stop_enable: Optional[bool] = True
    early_stop_patience: Optional[int] = 100
    early_stop_threshold: Optional[float] = 0.01


@dataclass
class LoggingConfig:
    enable_rerun: bool = True


@dataclass
class DatasetConfig:
    dataset_type: DatasetType = DatasetType.custom
    trajectory_reader: Optional[TrajectoryReaderConfig] = \
        field(default_factory=TrajectoryReaderConfig)
    cloud_reader: Optional[PointCloudReaderConfig] = \
        field(default_factory=PointCloudReaderConfig)


@dataclass
class OutputConfig:
    # Output main folder. Each experiment will be saved on a subfolder.
    folder: Optional[str] = None
    # Output trajectory format
    writer: TrajectoryWriterType = TrajectoryWriterType.tum


@dataclass
class PreprocessingConfig:
    image_height: int = 0
    image_width: int = 0
    depth_min: float = 0.0
    depth_max: float = 1e6
    enable_ground_segmentation: Optional[bool] = True


@dataclass
class OptimizationConfig:
    ...


@dataclass
class Configuration:
    inherit_from: Optional[str] = None
    data: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    opt: OptimizationConfig = field(default_factory=OptimizationConfig)


def load_configuration(filename: Path, cli_args: list[str] = None) -> \
        Configuration:
    """
    Recursively load a configuration file described by the Configuration
    dataclass.

    Args:
        filename: A path to a configuration file in yaml format.
        cli_args: A list of extra cli arguments to be merged in the
        final configuration.

    Returns:
    A Configuration object with initialized parameters.
    """
    default_cfg = OmegaConf.structured(Configuration)
    derived_cfg = OmegaConf.load(filename)
    if derived_cfg.get("inherit_from") is not None:
        logger.debug(f"Recursively loading configuration from {
                     derived_cfg.get('inherit_from')}")
        base_cfg = load_configuration(derived_cfg["inherit_from"])
        cfg = OmegaConf.merge(default_cfg, base_cfg, derived_cfg)
    else:
        cfg = OmegaConf.merge(default_cfg, derived_cfg)

    if cli_args is not None:
        override_cfg = OmegaConf.from_cli(cli_args)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def save_configuration(filename: Path, configuration: Configuration) \
        -> None:
    OmegaConf.save(configuration, filename)
