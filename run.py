import typer
from pathlib import Path
from utils.config_utils import load_configuration
from utils.logging_utils import get_logger, set_log_level
from typing_extensions import Annotated

app = typer.Typer()

logger = get_logger("main")


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
    set_log_level(verbose)
    logger.info("Running SLAM mode")
    cfg = load_configuration(configuration, ctx.args)
    logger.info(f"Running experiment with configuration:\n{cfg}")
    logger.error("Not implemented yet!")
    ...


@app.command("mesh",
             short_help="Generate a mesh of the environment from the "
             "SLAM output")
def mesh_main(input_model: Path, output_filename: Path):
    logger.error("Not implemented yet!")
    ...


@app.command("eval_trajectory",
             short_help="Evaluate the RPE of an estimated trajectory "
             "against the reference.")
def eval_trajectory(estimate_filename: Path, reference_filename: Path,
                    output_filename: Path):
    logger.error("Not implemented yet!")
    ...


@app.command("eval_mapping",
             short_help="Evaluate the mapping metrics "
             "(Accuracy, Completeness, Charmfer-L1, F1-score) "
             "of one or more estimated mesh against the reference point "
             "cloud.")
def eval_mapping(estimate_filename: list[Path], reference_filename: Path,
                 output_filename: Path):
    logger.error("Not implemented yet!")
    ...


@app.command("crop_reference_pcd",
             short_help="Crop the reference point cloud given a set of "
             "estimated meshes. Useful for multi-approach evaluation")
def crop_intersection(estimate_filenames: list[Path],
                      reference_filename: Path,
                      output_filename: Path):
    logger.error("Not implemented yet!")
    ...


if __name__ == "__main__":
    app()
