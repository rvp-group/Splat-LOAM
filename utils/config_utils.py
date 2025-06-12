from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from omegaconf import OmegaConf
from pathlib import Path
from utils.logging_utils import get_logger
from gsaligner import GSAlignerParams

logger = get_logger("")


class TrackingMethod(str, Enum):
    gt = "gt"
    gsaligner = "gsaligner"


class DatasetType(str, Enum):
    generic = "generic"
    vbr = "vbr"
    kitti = "kitti"
    ncd = "ncd"
    oxspires = "oxspires"
    oxspires_vilens = "oxspires_vilens"


class TrajectoryReaderType(str, Enum):
    kitti = "kitti"
    tum = "tum"
    vilens = "vilens"
    null = "null"


class TrajectoryWriterType(str, Enum):
    kitti = "kitti"
    tum = "tum"


class DataLoggerType(str, Enum):
    rerun = "rerun"
    wandb = "wandb"
    tensorboard = "tensorboard"


@dataclass
class TrajectoryReaderConfig:
    # Format of trajectory file
    reader_type: Optional[TrajectoryReaderType] = None
    # File containing pose information
    filename: Optional[str] = None
    # Association tolerance for timestamp (1ms by default)
    timestamp_dtol: float = 1e-3
    # If using KITTI, you can pass the times.txt file
    # to allow the loading of timestamps
    # kinda required for evaluation! (DatasetReader does this for you)
    timestamp_from_filename_kitti: Optional[str] = None
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
    method: TrackingMethod = TrackingMethod.gsaligner
    keyframe_threshold_distance: float = 1.0
    keyframe_threshold_nframes: int = -1
    keyframe_threshold_fitness: float = -1.0
    gsaligner: Optional[GSAlignerParams] = None


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
    opt_lambda_alpha: float = 1e-1
    opt_lambda_normal: float = 1e-1
    # Maximum per-axis scaling value
    opt_scaling_max: float = 0.5
    # Penalty for higher scaling factors
    opt_scaling_max_penalty: float = 0.2

    lmodel_threshold_ngaussians: Optional[int] = None
    lmodel_threshold_nkeyframes: Optional[int] = None


@dataclass
class LoggingConfig:
    enable: bool = True
    logger_type: Optional[DataLoggerType] = DataLoggerType.rerun
    # Concerning rerun, only one of the following options should be set.
    # If more are set, only the first will be used.
    # Spawn the rerun GUI
    rerun_spawn: Optional[bool] = True
    # should be enabled for remote viewer. Does not spawn a GUI but
    # serve log-data over gRPC
    rerun_serve_grpc: Optional[bool] = None
    # if a remote viewer is already instantiated, provide
    # url here to allow connection
    rerun_connect_grpc_url: Optional[str] = None


@dataclass
class DatasetConfig:
    dataset_type: DatasetType = DatasetType.generic
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
    """
    As mentioned in the paper, with depth, we really refer to
    point's ranges (norm([x,y,z]).
    """
    # Internal LiDAR image height (typically equal to vertical beams)
    image_height: int = 0
    # Internal LiDAR image width (typically equal to horizontal samples)
    image_width: int = 0
    # Minimum valid range
    depth_min: float = 0.0
    # Maximum valid range
    depth_max: float = 1e6
    enable_normal_estimation: Optional[bool] = True
    enable_ground_segmentation: Optional[bool] = True


@dataclass
class OptimizationConfig:
    position_lr: float = 0.0005
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    # depth_ratio indicates whether expected or median
    # depth should be rasterized
    # 0 -> expected
    # 1 -> median
    depth_ratio: float = 0


@dataclass
class Configuration:
    inherit_from: Optional[str] = None
    data: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(
        default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    opt: OptimizationConfig = field(default_factory=OptimizationConfig)
    device: str = "cuda:0"


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
        logger.debug(f"Recursively loading configuration from "
                     f"{derived_cfg.get('inherit_from')}")
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
    # Sanity check
    assert configuration == OmegaConf.load(filename)
