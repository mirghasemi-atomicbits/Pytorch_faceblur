#!/usr/bin/env bash

crop="crop=840:380:200:20"

ffmpeg -i videos/frame-original.jpg -vf $crop videos/frame-original-crop.jpg

for i in {1..9}; do
    ffmpeg -i videos/frame-str0$i.jpg -vf $crop videos/frame-crop-str0$i.jpg
done

ffmpeg -i videos/frame-str10.jpg -vf $crop videos/frame-crop-str10.jpg
