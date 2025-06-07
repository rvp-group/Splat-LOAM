import numpy as np
import open3d as o3d
from utils.config_utils import Configuration
from utils.logging_utils import get_logger
from scene.cameras import Camera
from scene.frame import Frame
import pyprojections as pyp


logger = get_logger("preprocessing")


class Preprocessor:
    """
    The Preprocessor class is responsible of converting the
    input point clouds to the appropriate format digestible
    by Splat-LOAM.
    """

    def __init__(self, cfg: Configuration):
        self.device = cfg.device
        self.cfg = cfg.preprocessing

    def __call__(self,
                 cloud: np.ndarray,
                 timestamp: float,
                 gt_pose: np.ndarray) -> Frame:
        """
        The function completes the following steps:
        - Compute the optimal cloud intrinsics.
        - Perform spherical projection
        - [Optionally] Segment ground points
        - [Optionally] Compute normals
        - Yield results in the form of a CameraInfo object.

        Args:
            cloud: [N, 3] np.float32 array
            timestamp: float
            gt_pose: [4, 4] np.float32 array
        """
        K, _, vfov, hfov = pyp.calculate_spherical_intrinsics(
            cloud.T, self.cfg.image_height, self.cfg.image_width
        )
        projector = pyp.Camera(
            self.cfg.image_height,
            self.cfg.image_width,
            K,
            self.cfg.depth_min,
            self.cfg.depth_max,
            pyp.CameraModel.Spherical
        )
        lut, _ = projector.project(cloud.T)
        invalid_mask = lut == -1
        valid_image = ~invalid_mask
        ranges = np.linalg.norm(cloud, axis=1)
        range_image = ranges[lut]
        range_image[invalid_mask] = 0.0
        normals = np.zeros_like(cloud, dtype=np.float32)
        valid_mask = (ranges > self.cfg.depth_min) & (
            ranges <= self.cfg.depth_max)
        normals[valid_mask] = self.compute_normals(cloud[valid_mask])
        normals_image = normals[lut]
        normals_image[invalid_mask] = np.float32([0, 0, 0])

        camera = Camera(
            K=K,
            image_depth=range_image[None, ...],
            image_normal=normals_image.transpose(2, 0, 1),
            image_valid=valid_image[None, ...],
            world_T_lidar=gt_pose)
        return Frame(
            camera=camera,
            timestamp=timestamp,
            device=self.device)

    def compute_normals(self, cloud: np.ndarray) -> np.ndarray:
        """
        The function provides normal information to the point cloud.
        By default, as mentioned in the paper (sec 3.3.1), we initialize
        the normals by directing them towards the sensor center to enhance
        initial visibility.
        However, if set, the function also:
        - computes normals in 3D space via PCA (Open3D)
        - segment and assign up-facing normals to the ground (pypatchworkpp)
        From our experiments, bootstrapping normals via PCA does not improve
        significantly the final result, while segmenting the ground does
        provide better results concerning the mapping results.
        We also found ground segmentation more effective and easier to tune
        w.r.t. the PCA estimation, thus we prioritize the former.
        """
        if self.cfg.enable_normal_estimation:
            # Convert the cloud in Open3D format and compute normals
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.5, max_nn=50
                ))
            pcd.orient_normals_towards_camera_location()
            normals = np.asarray(pcd.normals)
        else:
            normals = -cloud / np.linalg.norm(cloud, axis=1)[..., None]

        if self.cfg.enable_ground_segmentation:
            # TODO: Implement ground segmentation via patchworkpp
            raise NotImplementedError(
                "Ground segmentation still not implemented")
        return normals
