#!/bin/bash
# Expose the X server on the host.
sudo xhost +local:root
# --rm: Make the container ephemeral (delete on exit).
# -it: Interactive TTY.
# --gpus all: Expose all GPUs to the container.
# access jupyter notebook via http://localhost:1234/
docker run \
  --rm \
  -it \
  --gpus all \
  -v $(pwd):/nerf \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 1234:8888 -p 6006:6006 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  --name nerf_container nerf-dev