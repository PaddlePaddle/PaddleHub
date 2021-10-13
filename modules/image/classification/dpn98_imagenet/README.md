# dpn98_imagenet

|模型名称|dpn98_imagenet|
| :--- | :---: |
|类别|图像-图像分类|
|网络|DPN|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|是|
|模型大小|238MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：


- ### 模型介绍

  - DPN(Dual Path Networks) 是 ImageNet 2017 目标定位冠军的图像分类模型，DPN 融合了 ResNet 和 DenseNet 的核心思想。ResNet 通过跨层参数共享和保留中间特征的方式，可以有效地降低特征冗余度，重复利用已有特征，但缺点在于难以利用高层特征图再挖掘底层特征。DenseNet 的每一层都在之前所有层的输出中重新提取有用信息，可以有效地利用高层信息再次挖掘底层的新特征，但却存在特征上的冗余。DPN 有着以上两种拓扑路径的长处，可以共享公共特征，并通过双路径架构保留灵活性来探索新的特征。该PaddleHub Module结构为 DPN98，基于ImageNet-2012数据集训练，接受输入图片大小为 224 x 224 x 3，支持直接通过命令行或者Python接口进行预测。



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install dpn98_imagenet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run dpn98_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="dpn98_imagenet")
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

  - ```shell
    $ hub install dpn98_imagenet==1.0.0
    ```
