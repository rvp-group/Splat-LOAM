import typer
from pathlib import Path
import random
import numpy as np
from utils.config_utils import (
    load_configuration, Configuration, save_configuration,
    TrackingMethod)
from utils.logging_utils import get_logger, set_log_level
from typing_extensions import Annotated
from slam.slam import SLAM
from scene.dataset_readers import get_dataset_reader, DatasetReader
from scene.preprocessing import Preprocessor
from rich.progress import track
import torch
import rerun as rr


app = typer.Typer()

logger = get_logger("main")
set_log_level(0)


@app.command("slam",
             short_help="Run SLAM mode from a dataset",
             help="""Run SLAM mode from a dataset.
             The application runs SLAM over an input configuration
             file. Additional tunings can be set via CLI with the
             following format:

             python3 run.py slam <config_file> <param>.<subparam>=<value>



             i.e.

             python3 run.py slam configs/default.yaml
             mapping.num_iterations=200

             """,
             context_settings={"allow_extra_args": True,
                               "ignore_unknown_options": True})
def slam_main(ctx: typer.Context,
              configuration: Path,
              verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False
              ):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    rr.init("SplatLOAM")
    rr.serve_web(open_browser=False)
    set_log_level(verbose)
    cfg = load_configuration(configuration, ctx.args)
    logger.info(f"Running experiment with configuration:\n{cfg}")
    data_loader = get_dataset_reader(cfg)
    preprocessor = Preprocessor(cfg)
    slam_module = SLAM(cfg)
    pipeline_sanity_check(cfg, data_loader, preprocessor, slam_module)
    for cloud, timestamp, pose in track(data_loader,
                                        description="Processing frames"):
        frame = preprocessor(cloud, timestamp, pose)
        slam_module.process(frame)

    input("Done. Press Enter to finalize")

    raise NotImplementedError("Not yet done!")
    ...


@app.command("mesh",
             short_help="Generate a mesh of the environment from the "
             "SLAM output")
def mesh_main(input_model: Path, output_filename: Path):
    raise NotImplementedError("Not yet done!")
    ...


@app.command("eval_trajectory",
             short_help="Evaluate the RPE of an estimated trajectory "
             "against the reference.")
def eval_trajectory(estimate_filename: Path, reference_filename: Path,
                    output_filename: Path):
    raise NotImplementedError("Not yet done!")
    ...


@app.command("eval_mapping",
             short_help="Evaluate the mapping metrics "
             "(Accuracy, Completeness, Charmfer-L1, F1-score) "
             "of one or more estimated mesh against the reference point "
             "cloud.")
def eval_mapping(estimate_filename: list[Path], reference_filename: Path,
                 output_filename: Path):
    raise NotImplementedError("Not yet done!")
    ...


@app.command("crop_reference_pcd",
             short_help="Crop the reference point cloud given a set of "
             "estimated meshes. Useful for multi-approach evaluation")
def crop_intersection(estimate_filenames: list[Path],
                      reference_filename: Path,
                      output_filename: Path):
    raise NotImplementedError("Not yet done!")
    ...


@app.command("generate_dummy_cfg",
             short_help="Generate a configuration file in yaml format "
             "with default parameters setup")
def generate_dummy_cfg(output_filename: Path):
    cfg = Configuration()
    logger.info(f"Default cfg: {cfg}")
    logger.info(f"Saved at {output_filename}")
    save_configuration(output_filename, cfg)


def pipeline_sanity_check(cfg: Configuration,
                          data_loader: DatasetReader,
                          preprocessor: Preprocessor,
                          slam_module: SLAM) -> None:
    """
    Entrypoint to verify the post-initialization state of the system
    """
    from utils.trajectory_utils import TrajectoryReader_NULL
    # For mapping, verify that if tracking.method.gt is set,
    # TrajectoryReader is not null
    if cfg.tracking.method == TrackingMethod.gt and \
            isinstance(data_loader.traj_reader, TrajectoryReader_NULL):
        raise RuntimeError("Tracking method is set to GT but DatasetReader has"
                           " TrajectoryReader_NULL set. "
                           "Verify input trajectory file.")
    # Add other checks here before running the pipeline
    ...
    return


if __name__ == "__main__":
    app()
