from pytransform3d.rotations import (
    quaternion_from_matrix,
    quaternion_wxyz_from_xyzw,
    norm_matrix
)
import re
from pytransform3d.transformations import (transform_from_pq, check_transform)
import numpy as np
from pathlib import Path
from typing import List
from utils.config_utils import (
    TrajectoryReaderConfig
)


class TrajectoryReader:
    """
    Base class for handling trajectory data.
    The class contains different interfaces to extract sensor-aligned poses.
    Refer to explicit trajectory-type implementations for more details.
    """

    def __init__(self, config: TrajectoryReaderConfig):
        self.dtol = config.timestamp_dtol
        self.timestamps = []
        self.poses = []
        self.current_index = 0
        if config.gt_T_sensor_t_xyz_q_xyzw is not None:
            gt_T_s_pq = np.float32(config.gt_T_sensor_t_xyz_q_xyzw)
            gt_T_s_pq[3:] = quaternion_wxyz_from_xyzw(gt_T_s_pq[3:])
            self.gt_T_s = transform_from_pq(gt_T_s_pq)
        elif config.gt_T_sensor_kitti_filename is not None:
            raise RuntimeError(
                "Reading calib from kitti calibration file is not yet\
                    supported.")
        else:
            # Assuming gt_T_lidar as identity
            self.gt_T_s = np.eye(4)

    def __call__(self, timestamp: float, *args, **kwargs) -> np.ndarray:
        idx = self._find_closest_timestamp_idx(timestamp)
        return self.poses[idx] @ self.gt_T_s

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.current_index >= len(self.poses):
            raise StopIteration
        pose = self.poses[self.current_index] @ self.gt_T_s
        self.current_index += 1
        return pose

    def __getitem__(self, idx) -> np.ndarray:
        return self.poses[idx]

    def _find_closest_timestamp_idx(self, timestamp: float) -> int:
        closest_idx = min(
            range(len(self.timestamps)),
            key=lambda i: abs(self.timestamps[i] - timestamp)
        )
        if abs(self.timestamps[closest_idx] - timestamp) > self.dtol:
            raise RuntimeError(
                f"No timestamp found within tolerance {self.dtol}")
        return closest_idx


class TrajectoryReader_KITTI(TrajectoryReader):
    """
    Reads a trajectory file in KITTI format.
    Each row of the input file should be formatted as:
    P_00 P_01 P_02 P_03 P_10 P_11 P_12 P_13 P_20 P_21 P_22 P_23
    """

    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)
        self.index = 0
        with open(config.filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            pose_vect = np.array([float(x) for x in line.split()])
            pose = pose_vect.reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))
            self.poses.append(pose)

    def __call__(self, _: float, *args, **kwargs) -> np.ndarray:
        raise RuntimeError(
            "TrajectoryReader_KITTI does not allow random access")

    def _find_closest_timestamp_idx(self, _: float) -> int:
        raise RuntimeError(
            "TrajectoryReader_KITTI does not allow timestamped access")


class TrajectoryReader_TUM(TrajectoryReader):
    """
    Reads a trajectory file in TUM format.
    Each row of the input file should be formatted as:
    timestamp x y z q_x q_y q_z q_w
    """

    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)
        with open(config.filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            line = re.sub(" {2,}", " ", line)  # fix double spaces
            pose_vect = np.array([float(x) for x in re.split(" |, ", line)])
            self.timestamps.append(pose_vect[0])
            pq = pose_vect[1:]
            pq[3:] = quaternion_wxyz_from_xyzw(pq[3:])
            self.poses.append(transform_from_pq(pq))


class TrajectoryReader_VILENS(TrajectoryReader):
    """
    Reads a trajectory file in Vilens format
    (currently found on Oxspires dataset)
    Each row of the input file should be formatted as:
    counter, ts.secs, ts.nsecs, x, y, z, qx, qy, qz, qw
    """

    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)
        with open(config.filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            pose_vect = np.array([float(x) for x in re.split(" |, ", line)])
            self.timestamps.append(pose_vect[1] + pose_vect[2] / 1e9)
            pq = pose_vect[3:]
            pq[3:] = quaternion_wxyz_from_xyzw(pq[3:])
            self.poses.append(transform_from_pq(pq))


class TrajectoryReader_NULL(TrajectoryReader):
    """
    NULL trajectory, useful if no input ground truth is provided.
    The class always returns the identity matrix as pose.
    """

    def __init__(self, config: TrajectoryReaderConfig):
        TrajectoryReader.__init__(self, config)

    def __call__(self, _: float, *args, **kwargs) -> np.ndarray:
        return np.eye(4)

    def __iter__(self):
        return self

    def __next__(self):
        return np.eye(4)

    def __getitem__(self, idx):
        return np.eye(4)


trajectory_reader_available = {
    "kitti": TrajectoryReader_KITTI,
    "tum": TrajectoryReader_TUM,
    "vilens": TrajectoryReader_VILENS,
    "null": TrajectoryReader_NULL
}


class TrajectoryWriter_TUM:
    @staticmethod
    def write(filename: Path, poses: List[np.ndarray],
              timestamps: List[float]) -> None:
        """
        Writes a trajectory file in TUM format, provided a set of timestamped
        poses.
        Each line of the output file is formatted as:
        timestamp x y z q_x q_y q_z q_w
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            f.write("#timestamp tx ty tz qx qy qz qw\n")
        for timestamp, pose in zip(timestamps, poses):
            wtc = pose
            # fix transform matrix in case of numerical errors
            wtc[-1] = np.array([0, 0, 0, 1])
            wtc[:3, :3] = norm_matrix(wtc[:3, :3])
            wtc = check_transform(wtc)
            pq = np.hstack(
                (
                    wtc[:3, 3],
                    quaternion_from_matrix(wtc[:3, :3], strict_check=False),
                )
            )
            f.write(
                f"{timestamp:.6f} {pq[0]:.4f} {pq[1]:.4f} {
                    pq[2]:.4f} {pq[4]} {pq[5]} {pq[6]} {pq[3]}\n"
            )


class TrajectoryWriter_KITTI:
    @staticmethod
    def write(filename: Path, poses: List[np.ndarray],
              timestamps: List[float] = None) -> None:
        """
        Writes a trajectory file in KITTI format, provided a set of
        poses.
        Each line of the output file is formatted as:
        P_00 P_01 P_02 P_03 P_10 P_11 P_12 P_13 P_20 P_21 P_22 P_23
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            for pose in poses:
                wtc = pose
                # fix transform matrix in case of numerical errors
                wtc[-1] = np.array([0, 0, 0, 1])
                wtc[:3, :3] = norm_matrix(wtc[:3, :3])
                wtc = check_transform(wtc)
                f.write(
                    f"{wtc[0, 0]:.6f} {wtc[0, 1]:.6f} {wtc[0, 2]:.6f} \
                    {wtc[0, 3]:.6f} "
                    f"{wtc[1, 0]:.6f} {wtc[1, 1]:.6f} {wtc[1, 2]:.6f} \
                    {wtc[1, 3]:.6f} "
                    f"{wtc[2, 0]:.6f} {wtc[2, 1]:.6f} {wtc[2, 2]:.6f} \
                    {wtc[2, 3]:.6f}\n"
                )


trajectory_writer_available = {
    "tum": TrajectoryWriter_TUM,
    "kitti": TrajectoryWriter_KITTI
}
