FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel AS builder

ENV TORCH_CUDA_ARCH_LIST="8.6 8.9+PTX"

RUN apt update && apt install -y \
  git \
  libeigen3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY submodules /workspace

RUN pip install diff-surfel-spherical-rasterization/ simple-knn/
RUN pip install pyprojections==0.0.3

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
ARG UNAME=user
ARG UID=1000
ARG GID=1000
WORKDIR /workspace
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

RUN pip install rosbags typer[all] omegaconf matplotlib pytransform3d \
    plyfile natsort open3d rerun-sdk evo pyprojections

RUN chmod -R 777 /workspace
USER $UNAME

