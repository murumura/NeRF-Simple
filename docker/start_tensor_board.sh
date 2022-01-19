#!/bin/bash
docker exec \
  -it \
  nerf_container \
  tensorboard --logdir=./exp/exp_tag/lego/logs --host=0.0.0.0 --port=6006
