import numpy as np
from rich.progress import track, Progress
import open3d as o3d
from utils.logging_utils import get_logger
from evo.core import metrics
from typing import List
from evo.core.units import Unit
from evo.core import sync
from evo.core.filters import FilterException
from evo.core.trajectory import PoseTrajectory3D
from pathlib import Path

logger = get_logger("")


def evaluate_rpe(
    estimated_trajectory: List[np.ndarray],
    gt_trajectory: List[np.ndarray],
    timestamps: List[float],
    gt_timestamps: List[float],
    is_kitti: bool = False,
) -> float:
    est = PoseTrajectory3D(
        poses_se3=estimated_trajectory, timestamps=timestamps)
    if is_kitti:
        ref = PoseTrajectory3D(
            poses_se3=gt_trajectory[: len(
                estimated_trajectory)], timestamps=timestamps
        )
    else:
        ref = PoseTrajectory3D(poses_se3=gt_trajectory,
                               timestamps=gt_timestamps)
        ref, est = sync.associate_trajectories(ref, est, max_diff=0.05)

    percentages = [0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55]
    ref_length = min(ref.path_length, est.path_length)
    logger.info(f"Reference length: {ref_length:.3f} m, "
                f"Estimate length: {est.path_length:.3f} m")

    rpe_acc = None
    for perc in percentages:
        delta = ref_length * perc

        rpe_metric = metrics.RPE(
            pose_relation=metrics.PoseRelation.point_distance,
            delta=delta,
            rel_delta_tol=0.1,
            delta_unit=Unit.meters,
            all_pairs=True,
        )
        try:
            rpe_metric.process_data((ref, est))
        except FilterException as e:
            logger.warning(e)
            continue
        delta_error = rpe_metric.error / delta
        if rpe_acc is None:
            rpe_acc = delta_error
        else:
            rpe_acc = np.hstack((rpe_acc, delta_error))
        logger.debug(f"RPE {perc * 100:.2f}: "
                     f"{delta_error.mean().item() / delta:.8f}")

    return rpe_acc.mean().item(), rpe_acc.std().item()


def evaluate_recon(reference_filename: Path,
                   estimate_filename: Path,
                   down_sample_res: float = 0.02,
                   threshold: float = 0.2,
                   truncation_acc: float = 0.5,
                   truncation_com: float = 0.5,
                   gt_bbox_mask_on: bool = True,
                   mesh_sample_point: int = 10_000_000,
                   generate_error_map: bool = False,
                   progress: Progress | None = None):
    """Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction (should be mesh)
        file_trgt: file path of target (shoud be point cloud)
        down_sample_res: use voxel_downsample to uniformly sample mesh points
        threshold: distance threshold used to compute precision/recall
        truncation_acc: points whose nearest neighbor is farther than the distance would not be taken into account (take pred as reference)
        truncation_com: points whose nearest neighbor is farther than the distance would not be taken into account (take trgt as reference)
        gt_bbox_mask_on: use the bounding box of the trgt as a mask of the pred mesh
        mesh_sample_point: number of the sampling points from the mesh
    Returns:

    Returns:
        Dict of mesh metrics (chamfer distance, precision, recall, f1 score, etc.)
    """
    if generate_error_map:
        raise NotImplementedError("Error map not yet implemented.")

    logger.info(f"Opening triangle mesh at {estimate_filename}")
    estimate_mesh = o3d.io.read_triangle_mesh(estimate_filename)
    logger.info(f"Opening point cloud at {reference_filename}")
    reference_pcd = o3d.io.read_point_cloud(reference_filename)

    if gt_bbox_mask_on:
        logger.info("Filtering prediction outside the reference bounding box")
        reference_bbox = reference_pcd.get_axis_aligned_bounding_box()
        bmin = reference_bbox.get_min_bound()
        bmax = reference_bbox.get_max_bound()
        bmin[2] -= down_sample_res
        bmax[2] += down_sample_res
        reference_bbox = o3d.geometry.AxisAlignedBoundingBox(bmin, bmax)
        estimate_mesh.crop(reference_bbox)
    logger.info(f"Sampling {mesh_sample_point} points from estimate mesh")
    estimate_pcd = estimate_mesh.sample_points_uniformly(mesh_sample_point)

    if down_sample_res > 0:
        logger.info(f"Downsampling point clouds to {down_sample_res} m")
        prev_points_count = len(estimate_pcd.points)
        estimate_pcd = estimate_pcd.voxel_down_sample(down_sample_res)
        reference_pcd = reference_pcd.voxel_down_sample(down_sample_res)
        new_points_count = len(estimate_pcd.points)
        logger.info(f"Estimate pcd from "
                    f"{prev_points_count} to {new_points_count}")

    estimate_verts = np.asarray(estimate_pcd.points)
    reference_verts = np.asarray(reference_pcd.points)
    logger.info("Computing correspondences reference -> estimate")
    _, dist_p = nn_correspondance(
        reference_verts, estimate_verts, truncation_acc, True)
    logger.info("Computing correspondences estimate -> reference")
    _, dist_r = nn_correspondance(
        estimate_verts, reference_verts, truncation_com, False)

    if generate_error_map:
        # Should generate map here...
        raise NotImplementedError("Error map not yet implemented.")

    dist_p_mean = np.mean(dist_p).item()
    dist_r_mean = np.mean(dist_r).item()

    chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)

    precision = np.mean((dist_p < threshold).astype("float")).item() * 100.0
    recall = np.mean((dist_r < threshold).astype("float")).item() * 100.0
    fscore = 2 * precision * recall / (precision + recall)
    metrics = {
        "MAE_accuracy (cm)": dist_p_mean * 100,
        "MAE_completeness (cm)": dist_r_mean * 100,
        "Chamfer_L1 (cm)": chamfer_l1 * 100,
        "Precision [Accuracy] (%)": precision,
        "Recall [Completeness] (%)": recall,
        "F-score (%)": fscore,
        "Inlier_threshold (m)": threshold,  # evlaution setup
        "Outlier_truncation_acc (m)": truncation_acc,  # evlaution setup
        "Outlier_truncation_com (m)": truncation_com,  # evlaution setup
    }
    return metrics


def nn_correspondance(target_verts: np.ndarray,
                      source_verts: np.ndarray,
                      truncation_dist: float,
                      ignore_outliers: bool):
    """
    For each vertex in source_verts, find the nearest vertex in target_verts
    Args:
    target_verts: (Nx3) numpy array
    source_verts: (Nx3) numpy array
    truncation_dist: (float) points whose nn is farther than truncation_dist
    are not accounted unless ignore_outliers is set to True
    ignore_outliers: (bool) if set, includes points whose nn is farther
    than truncation_dist.
    Returns:
    ([indices], [distances])
    """

    indices = []
    distances = []

    if len(target_verts) == 0 or len(source_verts) == 0:
        logger.warning("Empty target_verts or source_verts."
                       "Cannot compute nearest-neighbors.")
        return indices, np.empty(0)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(
        target_verts)
    truncation_dist_square = truncation_dist**2

    kdtree = o3d.geometry.KDTreeFlann(target_pcd)
    for vert in track(source_verts, "Computing nearest-neighbors"):
        _, inds, dist_square = kdtree.search_knn_vector_3d(
            vert, 1)

        if dist_square[0] < truncation_dist_square:
            indices.append(inds[0])
            distances.append(np.sqrt(dist_square[0]))
        else:
            if not ignore_outliers:
                indices.append(inds[0])
                distances.append(truncation_dist)
    return indices, np.array(distances)


def crop_union(
        reference_filename: Path,
        estimate_filenames: list[Path],
        threshold_dist: float = 1.2,
        mesh_sample_point: int = 10_000_000) -> o3d.geometry.PointCloud:
    """Get the union of ground truth point cloud according to the intersection of the predicted
    mesh by different methods
    Args:
        reference_filename: file path of the ground truth
        (should be point cloud)
        estimate_filenames: a list of the paths of different methods'
        reconstruction (should be mesh)
        dist_thre: nearest neighbor distance threshold in meter
        mesh_sample_point: number of the sampling points from the mesh
    Returns:
        reference_crop : (o3d.geometry.PointCloud) union-cropped
        reference cloud
    """
    logger.info(f"Opening point cloud at {reference_filename}")
    reference_pcd = o3d.io.read_point_cloud(reference_filename)
    reference_verts = np.asarray(reference_pcd.points)

    threshold_dist_square = threshold_dist**2

    estimate_pcds = [
        o3d.io.read_triangle_mesh(f).sample_points_uniformly(mesh_sample_point)
        for f in track(estimate_filenames, "Sampling estimated meshes")
    ]

    estimate_merged_verts = np.vstack(
        [np.asarray(pcd.points) for pcd in estimate_pcds])

    logger.info("Building KDTree with sampled points")
    estimate_pcd = o3d.geometry.PointCloud()
    estimate_pcd.points = o3d.utility.Vector3dVector(estimate_merged_verts)

    kdtree = o3d.geometry.KDTreeFlann(estimate_pcd)
    near_mask = np.array(
        [
            kdtree.search_knn_vector_3d(pt, 1)[2][0] < threshold_dist_square
            for pt in track(reference_verts, "Computing nearest-neighbors")
        ]
    )

    reference_union_verts = reference_verts[near_mask]

    reference_crop = o3d.geometry.PointCloud()
    reference_crop.points = o3d.utility.Vector3dVector(reference_union_verts)
    return reference_crop
