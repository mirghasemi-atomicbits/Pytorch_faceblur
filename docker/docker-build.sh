#!/usr/bin/env bash

image_name="retinaface_faceblur"

cp ../requirements.txt .

docker rmi -f "$image_name"

docker build -t "$image_name" .

echo ""
echo "Download the RetinaFace weights Resnet50_Final.pth from"
echo "https://www.kaggle.com/keremt/retina-face/version/3"
echo "and put them in the ./weights folder."
echo ""
echo "The backbone model will be downloaded on first run."
echo ""
