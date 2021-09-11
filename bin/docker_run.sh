#!/usr/bin/env bash

image_name="retinaface_faceblur"
WORK=$(pwd)

# to enable the GPUs, use: docker run --gpus all ... 
docker run -it --rm -v $WORK:/work -v $WORK/.cache:/home/appuser/.cache "$image_name" "$@" 
