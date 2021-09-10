# Face Blur

A [PyTorch](https://pytorch.org/) implementation for blurring the faces on any image to maintain the privacy of the person.

The code is modified from Retinaface code by [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) which is originally developed on [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). 

### Contents
- [Installation](#installation)
- [Pre-Trained weights Download](#pre-trained)
- [Testing](#testing)
- [References](#references)

## Installation
##### Clone and install
1. git clone https://github.com/rahuja123/Pytorch_faceblur.git

2. Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.

3. Codes are based on Python 3

## Weights
You can find pretrain model and trained model [here](https://drive.google.com/drive/folders/1V3yXelGzg0Ce7uzLIuXEPjavZ4ghv2Ae?usp=sharing) or [here](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1). 

## Testing
1. python main.py


## References
- [Pytorch-RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
