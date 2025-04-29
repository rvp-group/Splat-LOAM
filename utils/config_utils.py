from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from omegaconf import OmegaConf
from pathlib import Path


class TrackingMethod(str, Enum):
    GT = "gt"
    POINT_TO_POINT = "p2point"
    POINT_TO_PLANE = "p2plane"
    GSALIGNER = "gsaligner"


class LoggingLevel(str, Enum):
    ALL = "notset"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


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
    log_level: LoggingLevel = LoggingLevel.INFO


@dataclass
class DatasetConfig:
    ...


@dataclass
class OutputConfig:
    folder: Optional[str]
    writer = None  # TODO


@dataclass
class OptimizationConfig:
    ...


@dataclass
class Configuration:
    inherit_from: Optional[str]
    data: DatasetConfig = DatasetConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()
    mapping: MappingConfig = MappingConfig()
    tracking: TrackingConfig = TrackingConfig()
    opt: OptimizationConfig = OptimizationConfig()


def load_configuration(filename: Path, cli_args: List[str] = None) -> \
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
        base_cfg = load_configuration(derived_cfg["inherit_from"])
        cfg = OmegaConf.merge(default_cfg, base_cfg, derived_cfg)
    else:
        cfg = OmegaConf.merge(default_cfg, base_cfg)

    if cli_args is not None:
        override_cfg = OmegaConf.from_cli(cli_args)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg
