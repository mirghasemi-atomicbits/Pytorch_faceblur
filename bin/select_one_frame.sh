#!/usr/bin/env bash

# ffmpeg -ss 00:00:00.782 -i videos/FRIENDS.mp4 -vframes 1 -q:v 2 videos/frame-original.jpg

# for i in {1..9}; do
#     ffmpeg -ss 00:00:00.782 -i videos/FRIENDS_blur-str0$i.mp4 -vframes 1 -q:v 2 videos/frame-str0$i.jpg
# done

ffmpeg -ss 00:00:00.782 -i videos/FRIENDS_blur-str05.mp4 -vframes 1 -q:v 2 videos/frame-str05.jpg

# see: https://stackoverflow.com/questions/27568254/how-to-extract-1-screenshot-for-a-video-with-ffmpeg-at-a-given-time
