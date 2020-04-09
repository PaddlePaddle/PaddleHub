# PaddleHub口罩人脸识别及分类模型C++预测部署

百度通过 `PaddleHub` 开源了业界首个口罩人脸检测及人类模型，该模型可以有效检测在密集人类区域中携带和未携带口罩的所有人脸，同时判断出是否有佩戴口罩。开发者可以通过 `PaddleHub` 快速体验模型效果、搭建在线服务，还可以导出模型集成到`Windows`和`Linux`等不同平台的`C++`开发项目中。

本文档主要介绍如何把模型在`Windows`和`Linux`上完成基于`C++`的预测部署。

主要包含两个步骤：
- [1. PaddleHub导出预测模型](#1-paddlehub导出预测模型)
- [2. C++预测部署编译](#2-c预测部署编译)

## 1. PaddleHub导出预测模型

#### 1.1 安装 `PaddlePaddle` 和 `PaddleHub`
  - `PaddlePaddle`的安装:
    请点击[官方安装文档](https://paddlepaddle.org.cn/install/quick) 选择适合的方式
  - `PaddleHub`的安装: `pip install paddlehub`

#### 1.2 从`PaddleHub`导出预测模型

在有网络访问条件下，执行`python export_model.py`导出两个可用于推理部署的口罩模型
其中`pyramidbox_lite_mobile_mask`为移动版模型, 模型更小，计算量低；
`pyramidbox_lite_server_mask`为服务器版模型，在此推荐该版本模型，精度相对移动版本更高。

成功执行代码后导出的模型路径结构：
```
pyramidbox_lite_server_mask
|
├── mask_detector   # 口罩人脸分类模型
|   ├── __model__   # 模型文件
│   └── __params__  # 参数文件
|
└── pyramidbox_lite # 口罩人脸检测模型
    ├── __model__   # 模型文件
    └── __params__  # 参数文件

```

## 2. C++预测部署编译

本项目支持在Windows和Linux上编译并部署C++项目，不同平台的编译请参考：
- [Linux 编译](./docs/linux_build.md)
- [Windows 使用 Visual Studio 2019编译](./docs/windows_build.md)
