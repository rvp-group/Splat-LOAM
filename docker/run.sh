if [ -n "$1" ]; then
  echo "Source dir: $1"
else
  echo "usage ./docker/.run.sh <SOURCE_DIR> <DATA_DIR>"
fi

if [ -n "$2" ]; then
  echo "Data dir: $2"
else
  echo "usage: ./docker/run.sh <SOURCE_DIR> <DATA_DIR>"
  exit -1
fi

IMAGE_NAME=splatloam

xhost +

docker run --gpus 'all' \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -ti \
  -it \
  --rm \
  --env="DISPLAY" \
  --shm-size 24G \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --privileged \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v "$2:/workspace/data/" \
  -v "$1:/workspace/repository/" \
  ${IMAGE_NAME} \
  bash -c /bin/bash
