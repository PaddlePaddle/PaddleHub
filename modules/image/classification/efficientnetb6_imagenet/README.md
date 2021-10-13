# efficientnetb6_imagenet

|模型名称|efficientnetb6_imagenet|
| :--- | :---: |
|类别|图像-图像分类|
|网络|EfficientNet|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|是|
|模型大小|170MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - EfficientNet 是谷歌的开源新模型，论文在 ICML 2019 发表。该模型从如何权衡网络的深度、宽度以及分辨率出发提出了复合扩展方法，使用了一个复合系数通过一种规范化的方式统一对网络的深度、宽度以及分辨率进行扩展。
EfficientNet 的基线网络是一个轻量级网络，它的主干网络由 MBConv 构成，同时采取了 squeeze-and-excitation 操作对网络结构进行优化。EfficientNet 系列模型先在小的基线网络使用网格搜索，然后直接使用不同的复合系数进行扩展，从而有效地减少了模型参数，提高了图像识别效率。该 PaddleHub Module结构为 EfficientNetB6，基于 ImageNet-2012 数据集训练，接受输入图片大小为 224 x 224 x 3，支持直接通过命令行或者 Python 接口进行预测。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install efficientnetb6_imagenet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run efficientnetb6_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="efficientnetb6_imagenet")
    test_img_path = "/PATH/TO/IMAGE"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    ```

- ### 3、API

  - ```python
    def classification(data)
    ```

    - **参数**
      - data：dict类型，key为image，str类型，value为待检测的图片路径，list类型。

    - **返回**
      - result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label对应的概率





## 四、更新历史

* 1.0.0

  初始发布

* 1.1.0
  - ```shell
    $ hub install efficientnetb6_imagenet==1.1.0
    ```
