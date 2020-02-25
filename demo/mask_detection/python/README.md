# 口罩佩戴检测模型Python高性能部署方案
百度通过 PaddleHub 开源了业界首个口罩人脸检测及人类模型，该模型可以有效检测在密集人类区域中携带和未携带口罩的所有人脸，同时判断出是否有佩戴口罩。开发者可以通过 PaddleHub 快速体验模型效果、搭建在线服务。

本文档主要介绍如何完成基于`python`的口罩佩戴检测预测。

主要包含两个步骤：
- [1. PaddleHub导出预测模型](#1-paddlehub导出预测模型)
- [2. 基于python的预测](#2-预测部署编译)

## 1. PaddleHub导出预测模型

#### 1.1 安装 `PaddlePaddle` 和 `PaddleHub`
  - `PaddlePaddle`的安装:
    请点击[官方安装文档](https://paddlepaddle.org.cn/install/quick) 选择适合的方式
  - `PaddleHub`的安装: `pip install paddlehub`
  - `opencv`的安装: `pip install opencv-python`
 
#### 1.2 安装`OpenCV` 相关依赖库
预测代码中需要使用`OpenCV`，所以还需要`OpenCV`安装相关的动态链接库。
`Ubuntu`下安装相关链接库：
```bash
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
```
CentOS 下安装相关链接库：
```bash
yum install -y libXext libSM libXrender
```

#### 1.2 从`PaddleHub`导出预测模型

在有网络访问条件下，执行`python export_model.py`导出两个可用于推理部署的口罩模型
其中`pyramidbox_lite_mobile_mask`为移动版模型, 模型更小，计算量低；
`pyramidbox_lite_server_mask`为服务器版模型，在此推荐该版本模型，精度相对移动版本更高。
成功执行代码后导出的模型路径结构：
```
pyramidbox_lite_mobile_mask
|
├── mask_detector   # 口罩人脸分类模型
|   ├── __model__   # 模型文件
│   └── __params__  # 参数文件
|
└── pyramidbox_lite # 口罩人脸检测模型
    ├── __model__   # 模型文件
    └── __params__  # 参数文件
```

## 2. 基于python的预测

### 2.1 
```
git clone https://github.com/PaddlePaddle/PaddleHub.git
cd PaddleHub/demo/mask_detection/python/
python export_model.py
```

### 2.1 执行预测程序
在终端输入以下命令进行预测:
```bash
python infer.py --models_dir=/path/to/models --img_paths=/path/to/images --video_path=/path/to/video --video_size=size/of/video --use_camera=(False/True)
--use_gpu=(False/True)
```
参数说明如下:

| 参数 | 是否必须|含义 |
|-------|-------|----------|
| models_dir | Yes|两个模型路径./pyramidbox_lite_mobile_mask |
| img_paths |No|需要预测的图片目录 |
| video_path |No|需要预测的视频目录|
| video_size |No|预测视频分辨率大小(w,h) |
| use_camera |No|是否打开摄像头进行预测 |
| use_gpu |No|是否GPU，默认为False|

##3. 可视化


