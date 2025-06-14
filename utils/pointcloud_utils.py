import numpy as np
import re
from utils.logging_utils import get_logger
import natsort
from pathlib import Path
from utils.config_utils import PointCloudReaderConfig, PointCloudReaderType
import open3d as o3d
from rosbags.highlevel import AnyReader
import sys
from typing import Iterable, List, Optional
from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2
from rosbags.typesys.types import sensor_msgs__msg__PointField as PointField

logger = get_logger("PointCloudReader")


class PointCloudReader():
    """
    Base class for handling point cloud data.
    This class provides a standard interface, refer to
    explicit format-type implementations for more details.
    """

    def __init__(self, config: PointCloudReaderConfig):
        self.n_clouds = 0
        self.current_index = 0

    def __len__(self):
        return self.n_clouds


class PointCloudReader_Collections(PointCloudReader):
    """
    This interface should be used for collection-type datasets.
    This class reads a collection of point clouds from a folder while
    parsing timestamps either from the filename or from a separate file.
    """

    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader.__init__(self, config)
        if config.timestamp_filename is not None:
            # Reads timestamps from a file
            self.timestamps = read_timestamps(config.timestamp_filename)
            self.get_timestamp = lambda x: self.timestamps[self.current_index]
        elif config.timestamp_from_filename:
            self.get_timestamp = lambda x: str_to_timestamp(x.stem)
        else:
            # Use a default timestamp of 0.0
            self.get_timestamp = lambda x: 0.0

    def __next__(self):
        if self.current_index >= self.n_clouds:
            raise StopIteration
        cloud = self.read_cloud(self.filenames[self.current_index])
        timestamp = self.get_timestamp(self.filenames[self.current_index])
        self.current_index += 1
        return cloud, timestamp

    def read_cloud(self, filename: Path):
        raise NotImplementedError(
            "Abstract class should not implement this method")


class PointCloudReader_BIN(PointCloudReader_Collections):
    """
    Reads a ordered sequence of .bin point clouds.
    By default, the reader assumes the KITTI binary format:
    - [x, y, z, intensity] float4
    Different formats can be used with the sole assumption that the
    first three per-point data are its coordinates [x, y, z]

    Returns the point cloud as a numpy array of shape (N, 3)
    """

    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader_Collections.__init__(self, config)
        self.filenames = sorted(Path(config.cloud_folder).glob("*.bin"))
        self.n_clouds = len(self.filenames)
        if config.bin_format:
            self.bin_format = config.bin_format
        else:
            self.bin_format = "<f4"

    def __iter__(self):
        return self

    def read_cloud(self, filename: Path) -> np.ndarray:
        cloud_xyzi = np.fromfile(filename, self.bin_format).reshape(-1, 4)
        return cloud_xyzi[..., :3]


class PointCloudReader_PLY(PointCloudReader_Collections):
    """
    Reads a ordered sequence of .ply point clouds.
    This reader uses Open3D read_point_cloud method to parse
    the point clouds.

    Returns the point cloud as a numpy array of shape (N, 3)
    """

    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader_Collections.__init__(self, config)
        self.filenames = sorted(Path(config.cloud_folder).glob("*.ply"))
        self.n_clouds = len(self.filenames)

    def __iter__(self):
        return self

    def read_cloud(self, filename: Path) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(filename)
        return np.asarray(pcd.points).astype(np.float32)


class PointCloudReader_PCD(PointCloudReader_Collections):
    """
    Reads a ordered sequence of .pcd point clouds.
    This reader uses Open3D read_point_cloud method to parse
    the point clouds.

    Returns the point cloud as a numpy array of shape (N, 3)
    """

    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader_Collections.__init__(self, config)
        self.filenames = sorted(Path(config.cloud_folder).glob("*.pcd"))
        self.n_clouds = len(self.filenames)
        logger.info(f"Found {self.n_clouds} pcd clouds")

    def __iter__(self):
        return self

    def read_cloud(self, filename: Path) -> np.ndarray:
        pcd = o3d.io.read_point_cloud(filename)
        return np.asarray(pcd.points).astype(np.float32)


class PointCloudReader_ROSBAG(PointCloudReader):
    """
    Reads a ordered sequence of point clouds from a rosbag file.
    This reader uses the AnyReader object from 'rosbags' package
    therefore, both ROS1 and ROS2 rosbags files should be supported.
    Currently, only ROS1 bag files have been tested.
    """

    def __init__(self, config: PointCloudReaderConfig):
        PointCloudReader.__init__(self, config)
        self.bag = None
        if Path(config.cloud_folder).is_file():
            logger.debug(f"Opening rosbag: {config.cloud_folder}")
            self.bag = AnyReader([Path(config.cloud_folder)])
        else:
            bag_filenames = natsort.natsorted(
                [bag for bag in list(Path(config.cloud_folder).glob("*.bag"))]
            )
            logger.debug(f"Opening rosbags: {bag_filenames}")
            self.bag = AnyReader(bag_filenames)

        self.bag.open()
        connections = [
            x for x in self.bag.connections if x.topic == config.rosbag_topic]
        if len(connections) == 0:
            avail_topics = {x.topic for x in self.bag.connections}
            logger.error(f"Topic {config.rosbag_topic} not available"
                         f"in {avail_topics}")
        self.n_clouds = self.bag.topics[config.rosbag_topic].msgcount
        self.cloud_loader = self.bag.messages(connections=connections)

    def __iter__(self):
        return self

    def __next__(self):
        conn, _, raw = next(self.cloud_loader)
        cloud_msg = self.bag.deserialize(raw, conn.msgtype)
        timestamp = float(cloud_msg.header.stamp.sec) + \
            float(cloud_msg.header.stamp.nanosec) / 1e9
        cloud = read_points(cloud_msg)
        xyz = np.vstack([cloud["x"], cloud["y"], cloud["z"]]).T
        return xyz, timestamp


pointcloud_reader_available = {
    PointCloudReaderType.bin: PointCloudReader_BIN,
    PointCloudReaderType.ply: PointCloudReader_PLY,
    PointCloudReaderType.pcd: PointCloudReader_PCD,
    PointCloudReaderType.rosbag: PointCloudReader_ROSBAG
}


def str_to_timestamp(timestamp_str: str) -> float:
    """Converts a string composed as
    <TEXT*>_<ts_seconds>.<ts_nanoseconds>_<TEXT*>.<EXT>
        to a float timestamp in seconds

    """
    num_str = re.findall(r'\d+', timestamp_str)
    if len(num_str) == 1:
        return float(num_str[0])
    elif len(num_str) == 2:
        return float(num_str[0]) + float(num_str[1]) / 1e9
    else:
        raise ValueError(f"Invalid timestamp {timestamp_str}")


def read_timestamps(filename: Path) -> List[float]:
    """
    Read timestamps from a file (usually from KITTI-like datasets)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    timestamps = [float(line.strip()) for line in lines]
    return timestamps


"""
This file is based on
https://github.com/ros2/common_interfaces/blob/4bac182a0a582b5e6b784d9fa9f0dabc1aca4d35/sensor_msgs_py/sensor_msgs_py/point_cloud2.py
All rights reserved to the original authors: Tim Field and Florian Vahl.
"""

_DATATYPES = {}
_DATATYPES[PointField.INT8] = np.dtype(np.int8)
_DATATYPES[PointField.UINT8] = np.dtype(np.uint8)
_DATATYPES[PointField.INT16] = np.dtype(np.int16)
_DATATYPES[PointField.UINT16] = np.dtype(np.uint16)
_DATATYPES[PointField.INT32] = np.dtype(np.int32)
_DATATYPES[PointField.UINT32] = np.dtype(np.uint32)
_DATATYPES[PointField.FLOAT32] = np.dtype(np.float32)
_DATATYPES[PointField.FLOAT64] = np.dtype(np.float64)

DUMMY_FIELD_PREFIX = "unnamed_field"


class PointCloudXf:
    def __init__(self):
        self.fields: List[PointField] = []
        self.points: np.ndarray[None, np.dtype[np.float32]] = np.zeros(0)
        ...

    @staticmethod
    def from_ros(msg: PointCloud2):
        cloud = PointCloudXf()
        cloud.fields = msg.fields
        field_names = [f.name for f in msg.fields]
        cloud.points = read_points(msg, field_names)

        return cloud


def read_points(
        cloud: PointCloud2,
        field_names: Optional[List[str]] = None,
        uvs: Optional[Iterable] = None,
        reshape_organized_cloud: bool = False,
) -> np.ndarray:
    """
    Read points from a sensor_msgs.PointCloud2 message.
    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: None)
    :param reshape_organized_cloud: Returns the array as an 2D organized
            point cloud if set.
    :return: Structured NumPy array containing all points.
    """
    # Cast bytes to numpy array
    points = np.ndarray(
        shape=(cloud.width * cloud.height,),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data,
    )

    # Keep only the requested fields
    if field_names is not None:
        assert all(
            field_name in points.dtype.names for field_name in field_names
        ), "Requests field is not in the fields of the PointCloud!"
        # Mask fields
        points = points[list(field_names)]

    # Swap array if byte order does not match
    if bool(sys.byteorder != "little") != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)

    # Select points indexed by the uvs field
    if uvs is not None:
        # Don't convert to numpy array if it is already one
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        # Index requested points
        points = points[uvs]

    # Cast into 2d array if cloud is 'organized'
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)

    return points


def dtype_from_fields(fields: Iterable[PointField],
                      point_step: Optional[int] = None) -> np.dtype:
    """
    Convert a Iterable of sensor_msgs.msg.PointField messages to a np.dtype.
    :param fields: The point cloud fields.
                   (Type: iterable of sensor_msgs.msg.PointField)
    :param point_step: Point step size in bytes. Calculated from the given
            fields by default.
                       (Type: optional of integer)
    :returns: NumPy datatype
    """
    # Create a lists containing the names, offsets and datatypes of all fields
    field_names = []
    field_offsets = []
    field_datatypes = []
    for i, field in enumerate(fields):
        # Datatype as numpy datatype
        datatype = _DATATYPES[field.datatype]
        # Name field
        if field.name == "":
            name = f"{DUMMY_FIELD_PREFIX}_{i}"
        else:
            name = field.name
        # Handle fields with count > 1 by creating subfields with a suffix
        # consiting of "_" followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            if field.count > 1:
                subfield_name = f"{name}_{a}"
            else:
                subfield_name = name
            assert subfield_name not in field_names, \
                "Duplicate field names are not allowed!"
            field_names.append(subfield_name)
            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)

    # Create dtype
    dtype_dict = {"names": field_names,
                  "formats": field_datatypes, "offsets": field_offsets}
    if point_step is not None:
        dtype_dict["itemsize"] = point_step
    return np.dtype(dtype_dict)
