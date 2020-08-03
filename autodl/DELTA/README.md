# DELTA: DEep Learning Transfer using Feature Map with Attention for Convolutional Networks

## Introduction

This page implements the [DELTA](https://arxiv.org/abs/1901.09229) algorithm in [PaddlePaddle](https://www.paddlepaddle.org.cn).

> Li, Xingjian, et al. "DELTA: Deep learning transfer using feature map with attention for convolutional networks." ICLR 2019.

## Preparation of Data and Pre-trained Model

- Download transfer learning target datasets, like [Caltech-256](https://www.kaggle.com/jessicali9530/caltech256), [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) or others. Arrange the dataset in this way:
```
    root/train/dog/xxy.jpg
    root/train/dog/xxz.jpg
    ...
    root/train/cat/nsdf3.jpg
    root/train/cat/asd932_.jpg
    ...

    root/test/dog/xxx.jpg
    ...
    root/test/cat/123.jpg
    ...
```

- Download [the pretrained models](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/image_classification#resnet-series). We give the results of ResNet-101 below.

## Running Scripts

Modify `global_data_path` in `datasets/data_path` to the path root where the dataset is.

```bash
python -u main.py --dataset Caltech30 --delta_reg 0.1 --wd_rate 1e-4 --batch_size 64 --outdir outdir --num_epoch 100 --use_cuda 0
python -u main.py --dataset CUB_200_2011 --delta_reg 0.1 --wd_rate 1e-4 --batch_size 64 --outdir outdir --num_epoch 100 --use_cuda 0
```

Those scripts give the results below:

\ | l2 | delta
---|---|---
Caltech-256|79.86|84.71
CUB_200|77.41|80.05
