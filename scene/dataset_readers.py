from utils.pointcloud_utils import pointcloud_reader_available
import numpy as np
import copy
from utils.trajectory_utils import trajectory_reader_available
from pathlib import Path
from utils.config_utils import (
    Configuration, DatasetType)
from utils.pointcloud_utils import (
    PointCloudReader,
    PointCloudReader_BIN,
    PointCloudReader_PCD,
    PointCloudReader_ROSBAG
)
from utils.trajectory_utils import (
    TrajectoryReader,
    TrajectoryReader_KITTI,
    TrajectoryReader_TUM,
    TrajectoryReader_VILENS,
    TrajectoryReader_NULL
)
from utils.logging_utils import get_logger

logger = get_logger("")


class DatasetReader:
    """
    Base Dataset reader class.
    DatasetReader is an abstraction over the PointCloudReader
    and TrajectoryReader, allowing easier interaction
    over established datasets.
    Each dataset type should have its own DatasetReader.

    When queried, the reader returns a tuple containing:
        (cloud, timestamp, associated_gt_pose)
    """

    def __init__(self, cfg: Configuration):
        self.cfg = cfg
        self.cloud_reader: PointCloudReader = None
        self.traj_reader: TrajectoryReader = None

    def __next__(self):
        while True:
            cloud, timestamp = next(self.cloud_reader)

            try:
                gt_pose = self.traj_reader(timestamp)
                return cloud, timestamp, gt_pose

            except RuntimeError as e:
                if self.cfg.data.skip_clouds_wno_sync is True:
                    logger.warning(
                        f"{e} | Skipping unsynchronized cloud at {timestamp}")
                else:
                    logger.warning(
                        f"{e} | Setting gt_pose as identity")
                    return cloud, timestamp, np.eye(4)

        cloud, timestamp = next(self.cloud_reader)
        try:
            gt_pose = self.traj_reader(timestamp)
        except RuntimeError as e:
            self.cfg.data.skip_clouds_wno_sync
            logger.warning(f"{e} | Setting gt_pose as identity")
            gt_pose = np.eye(4)
        return cloud, timestamp, gt_pose

    def __len__(self):
        return len(self.cloud_reader)


class DatasetReader_KITTI(DatasetReader):
    """
    Reader for KITTI dataset.
    To use this reader, set the
    data.cloud_reader.cloud_folder to the base folder of
    a kitti sequence (i.e. folder containing velodyne and times.txt) and,
    if available, the data.trajectory_reader.filename to the corresponding
    ground truth trajectory file.

    When queried, the reader returns a tuple containing:
        (cloud, timestamp, associated_gt_pose)
    """

    def __init__(self, config: Configuration):
        DatasetReader.__init__(self, config)
        pc_cfg = config.data.cloud_reader
        base_folder = Path(pc_cfg.cloud_folder)
        if "velodyne" in base_folder.name:
            pc_cfg.timestamp_filename = base_folder.parent / "times.txt"
        else:
            pc_cfg.cloud_folder = base_folder / "velodyne"
            pc_cfg.timestamp_filename = base_folder / "times.txt"
        self.cloud_reader = PointCloudReader_BIN(pc_cfg)
        tr_cfg = config.data.trajectory_reader
        tr_cfg.gt_T_sensor_kitti_filename = base_folder / "calib.txt"
        if tr_cfg.filename is None or (not Path(tr_cfg.filename).is_file()):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            if tr_cfg.timestamp_from_filename_kitti is None:
                tr_cfg.timestamp_from_filename_kitti = \
                    pc_cfg.timestamp_filename
            self.traj_reader = TrajectoryReader_KITTI(tr_cfg)

    def __iter__(self):
        return self

    def __next__(self):
        cloud, timestamp = next(self.cloud_reader)
        gt_pose = next(self.traj_reader)
        return cloud, timestamp, gt_pose


class DatasetReader_VBR(DatasetReader):
    """
    Reader for the VBR dataset.
    To use this reader, set the
    data.cloud_reader.cloud_folder to the base folder of
    a VBR sequence (i.e. pincio_train0/) and, if available,
    the data.trajectory_reader.filename to the corresponding
    ground truth trajectory file.

    By default, the rosbag topic is set to /ouster/points

    By default, the gt_T_lidar_t_xyz_q_xyzw is set to
    [0, 0, 0, 0, 0, 0, 1]
    according to:
    https://rvp-group.net/slam-dataset.html

    When queried, the reader returns a tuple containing:
        (cloud, timestamp, associated_gt_pose)
    """

    def __init__(self, config: Configuration):
        DatasetReader.__init__(self, config)

        pc_cfg = config.data.cloud_reader
        if pc_cfg.rosbag_topic is None:
            pc_cfg.rosbag_topic = "/ouster/points"
        self.cloud_reader = PointCloudReader_ROSBAG(pc_cfg)

        tr_cfg = config.data.trajectory_reader
        tr_cfg.gt_T_sensor_t_xyz_q_xyzw = [0, 0, 0, 0, 0, 0, 1]
        if tr_cfg.filename is None or (not Path(tr_cfg.filename).is_file()):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            self.traj_reader = TrajectoryReader_TUM(tr_cfg)

    def __iter__(self):
        return self


class DatasetReader_NCD(DatasetReader):
    """
    Reader for the Newer College Dataset.
    This reader is made for the rosbag version of the
    dataset.
    To use this reader, set the
    data.cloud_reader.cloud_folder to the path of the rosbag
    or to the folder containing multiple bags and, if
    available, the
    data.trajectory_reader.filename to the corresponding
    ground truth trajectory file.

    By default, the rosbag topic is set to /os_cloud_node/points

    By default, the gt_T_lidar_t_xyz_q_xyzw is set to
    [0.001, 0, 0.091, 0, 0, 0, 1]
    according to:
    https://github.com/ori-drs/halo_description/blob/master/urdf/halo.urdf.xacro


    When queried, the reader returns a tuple containing:
        (cloud, timestamp, associated_gt_pose)
    """

    def __init__(self, config: Configuration):
        DatasetReader.__init__(self, config)

        pc_cfg = config.data.cloud_reader
        if pc_cfg.rosbag_topic is None:
            pc_cfg.rosbag_topic = "/os_cloud_node/points"
        self.cloud_reader = PointCloudReader_ROSBAG(pc_cfg)

        tr_cfg = config.data.trajectory_reader
        tr_cfg.gt_T_sensor_t_xyz_q_xyzw = [0.001, 0, 0.091, 0, 0, 0, 1]
        if tr_cfg.filename is None or (not Path(tr_cfg.filename).is_file()):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            self.traj_reader = TrajectoryReader_TUM(tr_cfg)

    def __iter__(self):
        return self


class DatasetReader_OXSPIRES(DatasetReader):
    """
    Reader for the Oxford Spires Dataset.
    This reader is made for the rosbag version of the dataset
    with the original ground truth trajectories in TUM format.
    To use this reader, set the
    data.cloud_reader.cloud_folder to the path of the rosbag
    or to the folder containing multiple bags and, if
    available, the
    data.trajectory_reader.filename to the corresponding
    ground truth trajectory file.

    By default, the rosbag topic is set to /hesai/pandar

    By default, the gt_T_lidar_t_xyz_q_xyzw is set to
    [0, 0, 0.124, 0, 0, 1, 0]
    according to:
    https://github.com/ori-drs/oxford_spires_dataset/blob/main/config/sensor.yaml

    When queried, the reader returns a tuple containing:
        (cloud, timestamp, associated_gt_pose)
    """

    def __init__(self, config: Configuration):
        DatasetReader.__init__(self, config)

        pc_cfg = config.data.cloud_reader
        if pc_cfg.rosbag_topic is None:
            pc_cfg.rosbag_topic = "/hesai/pandar"
        self.cloud_reader = PointCloudReader_ROSBAG(pc_cfg)

        tr_cfg = config.data.trajectory_reader
        tr_cfg.gt_T_sensor_t_xyz_q_xyzw = [0, 0, 0.124, 0, 0, 1, 0]
        if tr_cfg.filename is None or (not Path(tr_cfg.filename).is_file()):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            self.traj_reader = TrajectoryReader_TUM(tr_cfg)

    def __iter__(self):
        return self


class DatasetReader_OXSPIRES_VILENS(DatasetReader):
    """
    Reader for the Oxford Spires Dataset.
    This reader is made for the VILENS estimate version of the dataset.
    To use this reader, set the
    data.cloud_reader.cloud_folder to the path containing the pcd
    point clouds. The files should respect the naming convention of:
    cloud_<sec>_<nanosec>.pcd

    and, if available, set the
    data.trajectory_reader.filename to the corresponding
    ground truth trajectory file.

    By default, the gt_T_lidar_t_xyz_q_xyzw is set to
    [0, 0, 0.124, 0, 0, 1, 0]
    according to:
    https://github.com/ori-drs/oxford_spires_dataset/blob/main/config/sensor.yaml

    When queried, the reader returns a tuple containing:
        (cloud, timestamp, associated_gt_pose)
    """

    def __init__(self, config: Configuration):
        DatasetReader.__init__(self, config)

        pc_cfg = config.data.cloud_reader
        pc_cfg.timestamp_from_filename = True
        self.cloud_reader = PointCloudReader_PCD(pc_cfg)

        tr_cfg = config.data.trajectory_reader
        tr_cfg.gt_T_sensor_t_xyz_q_xyzw = [0, 0, 0, 0, 0, 0, 1]
        if tr_cfg.filename is None or (not Path(tr_cfg.filename).is_file()):
            self.traj_reader = TrajectoryReader_NULL(tr_cfg)
        else:
            self.traj_reader = TrajectoryReader_VILENS(tr_cfg)

    def __iter__(self):
        return self


class DatasetReader_GENERIC(DatasetReader):
    """
    Generic Reader for non-registered datasets.
    This class allows to mix any point cloud readers with any trajectory
    reader.
    For additional information, refer to the appropriate classes.

    When queried, the reader returns a tuple containing:
        (cloud, timestamp, associated_gt_pose)
    """

    def __init__(self, config: Configuration):
        DatasetReader.__init__(self, config)
        pc_cfg = config.data.cloud_reader
        tr_cfg = config.data.trajectory_reader

        self.cloud_reader = pointcloud_reader_available[pc_cfg.bin_format](
            pc_cfg)
        self.traj_reader = trajectory_reader_available[tr_cfg.reader_type](
            tr_cfg)

    def __iter__(self):
        return self


datasetreader_available = {
    DatasetType.vbr: DatasetReader_VBR,
    DatasetType.kitti: DatasetReader_KITTI,
    DatasetType.ncd: DatasetReader_NCD,
    DatasetType.oxspires: DatasetReader_OXSPIRES,
    DatasetType.oxspires_vilens: DatasetReader_OXSPIRES_VILENS,
    DatasetType.generic: DatasetReader_GENERIC
}


def get_dataset_reader(cfg: Configuration) -> DatasetReader:
    dr_type = cfg.data.dataset_type
    return datasetreader_available[dr_type](cfg)
