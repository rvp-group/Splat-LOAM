<p align="center">
  <h2 align="center">Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping</h2>
  <p align="center">
    <strong>Emanuele Giacomini</strong>
    ·
    <strong>Luca Di Giammarino</strong>
    ·
    <strong>Lorenzo De Rebotti</strong>
    ·
    <strong>Giorgio Grisetti</strong>
    ·
    <strong>Martin R. Oswald</strong>
  </p>
</p>
<h3 align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-2503.17491-b31b1b.svg)](https://arxiv.org/abs/2503.17491)

<p align="center">
  <img src="https://github.com/user-attachments/assets/6aee97f1-ea4a-4b56-bc50-50bae9e7d6c5"/>
</p>

# Quickstart
## Installation
### Prerequisites
First, download this repository and all its submodules:

```sh
git clone --recursive https://github.com/rvp-group/Splat-LOAM.git
  ```

To run Splat-LOAM, you strictly need an NVIDIA GPU with CUDA capabilities.
The implementation has been tested with CUDA versions `11.8` and `12.6`, however, other versions should be compatible

Moreover, depending on how you intend to run the implementation, you might need:
- **Docker** and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 
- [Pixi](https://pixi.sh/latest/installation/)

### 1. Recommended (Docker)
The Docker configuration can be used both for execution and for development and its definitely the less-painful way to setup your machine.

To build the Docker image, simply run
```sh
./docker/build.sh
```

Once installed, we can run the container:
```sh
./docker/run.sh . <YOUR_DATA_FOLDER>
```
*The first argument should point to the root project directory while the second argument should be the absolute path to your data.*

While in the container, everything you need will be found in the `/workspace` folder. Specifically:
- `/workspace/repository/` is a shared folder that points to the host repository (changes here are preserved)
- `/workspace/data/` is a shared folder that points to`<YOUR_DATA_FOLDER>`

*For who's interested in development: you can change code in the host repository without restarting the container which is quite handy. We still need a way to make LSP work, if you have any tips, get in contact with us!*
### 2. Pixi
>[!NOTE]
> Pixi configuration uses CUDA 12.6 while the Docker configuration uses CUDA 11.8. It shouldn't create any issues but its worth noticing this. 

Assuming that you already installed Pixi, simply run
```sh
pixi install
```

Once the environment is setup, we suggest starting a shell in the Pixi environment to continue forward:
```sh
pixi shell
```
## Run Splat-LOAM
The main entrypoint for Splat-LOAM is the `run.py` script. It provides the following commands:
* `slam`: Run SLAM mode from a dataset
* `mesh`: Extract a mesh from the SLAM output
* `eval_odom`: Evaluate the RPE of an estimated trajectory against the reference.
* `eval_recon`: Evaluate the 3D Reconstruction metrics (Accuracy, Completeness, Charmfer-L1, F1-score) of one or more estimated mesh against the reference point cloud.
* `crop_recon`: Crop the reference point cloud given a set of estimated meshes. Useful for multi-approach evaluation
* `generate_dummy_cfg`: Generate a configuration file in yaml format with default parameters setup

To run the slam application, we first need a configuration file!
### 1. Configuration

> [!TIP]
> **TL;DR**
>
> The `configs/` folder contains some configuration files that you can use to start.
>
> We are still working on porting the configurations to the repository. Odometry configurations in car-like scenarios are still not working correctly!
>
> **Just remember to change the data paths!**

<details>
    <summary>[Details (click to expand)]</summary>
A configuration is a YAML file that contains several info required for the SLAM application to run correctly. From a high level perspective, it is composed of the following components:

```yaml
Configuration:
  inherit_from : str  # base configuration path (default to null)
  data: DatasetConfig # Input data, format, location etc
  preprocessing: PreprocessingConfig # From PointCloud to Images
  output: OutputConfig # Output data, format, location etc
  logging: LoggingConfig # Logging backends
  mapping: MappingConfig # How Gaussians are updated
  tracking: TrackingConfig # How new poses are estimated
  opt: OptimizationConfig # GS learning rates
```

You don't need to write the full configuration unless you want to change everything. If you wish, you can also start from another configuration by filling the `inherit_from` parameter:
```yaml
inherit_from: configs/a_cool_config.cfg # paths can be relative from project root directory
```

If you're reading this, it's likely that you want to pass your data to Splat-LOAM so let's see how you can do it.

First, in our context, an input sequence should provide:

  1) **Point Clouds** (timestamped)
  2) (Optionally) **Poses** (also timestamped)

To maximize compatibility with different datasets, we rely on three main abstraction entities to parse input data:

  - `PointCloudReader` : Given data in *any* format, provides `<Point Cloud, timestamp>` elements
  - `TrajectoryReader` : Given a trajectory in *any* format, provide `f(Point Cloud, timestamp) -> pose`
  - `DatasetReader` : Given data in *any* format, provides iterable for `<Point Cloud, timestamp, pose>`

Since publicly available datasets follows more or less similar patterns and formats, once the `DatasetReader` object is informed of the dataset type, it will setup both `PointCloudReader` and `TrajectoryReader` to handle the underlying data.

Here we show the current set of supported dataset formats:

| DatasetReader   | Notes                     | Cloud Format          |      Trajectory Reader  |
|:---------------:|---------------------------|:---------------------:|:-----------------------:|
| vbr             | Vision Benchmark in Rome  |rosbag                 | tum                     |
| kitti           |                           |bin                    | kitti                   |
| ncd             | Newer College Dataset     |rosbag                 | tum                     |
| oxspires        | Oxford Spires             |rosbag                 | tum                     |
| oxpires-vilens  | w/ VILENS trajectory      |pcd                    | vilens                  |
| generic         | Customizable              |bin\|ply\|pcd\|rosbag  | kitti \| tum \| vilens  |

Suppose I want to set up a configuration to read the popular `quad-easy` sequence of NCD. I'd simply write over a new configuration file:
```yaml
data:
  dataset_type: ncd
  cloud_reader:
    cloud_folder: /path/to/ncd.bag
  trajectory_reader:
    filename: /path/to/trajectory.tum
```
And it's done. As long as the data remains consistent, the `DatasetReader` will handle the extra parameters (extrinsics, rosbag topics, formats, etc).

>GT trajectory is optional unless `tracking.method=gt` is set. If available, it's used for initial guess alignment only and for evaluation metadata retrival.
>
> More details on what each DatasetReader does, can be found in `scene.dataset_reader.py`

More details will be provided in the documentation (WIP)


</details>

### Running SLAM
To launch the SLAM application, run:
```sh
  python3 run.py slam <path/to/config.yaml>
```

If `output.folder` is not set, the experiment results will be placed in `results/<date_of_the_experiment>/`.

>[!TIP]
>If you want to solve `Mapping-only`, provide a trajectory in `data.trajectory_reader.filename`, set tracking to use it with `tracking.method=gt` and enable skipping of clouds that have no associated pose with `data.skip_clouds_wno_sync=true`

<details>
    <summary>[Details (click to expand)]</summary>

#### Logging
We use two logging systems within Splat-LOAM. For events, we use the `logging` module. If you want to enable debug logs, include the `--verbose / -v` argument.
For data, we rely on `rerun`. Specifically, at each frame, we log rasterized depths and normals and depth_L1 error map. We additionally log the densification mask and the current Gaussian model state.

You can customize the behavior of rerun through the configuration file:
  ```yaml
logging:
    enable: true/false # Enable data logging

    EITHER
    rerun_spawn: true/false/none # Spawn GUI and binds to it <use if monitor is available>
    OR
    rerun_serve_grpc: true/false/none # serve log-data over gRPC <use for remote connections>
    OR
    rerun_connect_grpc_url: <str> # serve log-data over gRPC to an already instantiated viewer.
  ```

> The Gaussian model observed in rerun is not rendered with the 2DGS rasterizer. Don't worry if it looks different from the rasterized images.

</details>

## Mesh generation
After running SLAM, you can generate a mesh representation by running:
```sh
python3 run.py mesh <path/to/result/folder>
```

>[!IMPORTANT]
>The default behavior of the mesh application is a trade-off between quality and speed. Checkout the documentation to tune the meshing process.
## Evaluation

### 3D Reconstruction
To evaluate the computed mesh, run the following:

```sh
python3 run.py eval_recon <reference_pointcloud_file> <estimate_mesh> 
```

By default, the script will provide the results both on the terminal and on the file `eval_recon_<date_of_exp>` located in the working directory.

>[!IMPORTANT]
>If the evaluation is needed for comparison with other methods, generate the cropped reference point cloud to ensure a fair comparison.
>This is achieved by running 
> 
> `python3 run.py crop_recon <reference_pointcloud_file> <mesh_1> <mesh_2> ... <mesh_n>`

### Odometry
If ground truth trajectory was provided in the experiment's configuration file, then you can evaluate the odometry estimate by running:

```sh
python3 run.py eval_odom <path/to/result/folder>
```

If no trajectory was provided, then a few more arguments are required:

```sh
python3 run.py eval_odom <path/to/odom/estimate> \
                          --reference-filename <path/to/reference/trajectory> \
                          --reference-format <tum|kitti|vilens> \
                          --estimate-format <tum|kitti|vilens> \
```

>[!NOTE]
>If reference is expressed in KITTI format, you also need to provide the argument:
>
>`--kitti-timestamps <path/to/kitti/times.txt>`
>
> Additionally, for KITTI sequences, we do not evaluate if number of poses between reference and estimate doesn't match.

# Citation
If you use Splat-LOAM for your research, please cite (currently in preprint):
```
@misc{giacomini2025splatloam,
      title={Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping}, 
      author={Emanuele Giacomini and Luca Di Giammarino and Lorenzo De Rebotti and Giorgio Grisetti and Martin R. Oswald},
      year={2025},
      eprint={2503.17491},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.17491}, 
}
```
# Contacts
If you have questions or suggestions, we invite you to open an issue!
Additionally, you can contact:

- Emanuele Giacomini : `giacomini@diag.uniroma1.it`
