import numpy as np
from rich.progress import track
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
