# resnext101_vd_64x4d_imagenet

|模型名称|resnext101_vd_64x4d_imagenet|
| :--- | :---: |
|类别|图像-图像分类|
|网络|ResNeXt_vd|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|是|
|模型大小|172MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：


- ### 模型介绍

  - ResNeXt 是由 UC San Diego 和 Facebook AI 研究所于2017年提出的图像分类模型。沿袭了 VGG/ResNets 的堆叠思想，ResNeXt 使用相同形状的块来加深网络，并且像
 Inception 网络一样，采用 split-transform-merge 策略来增加网络的分支数，但 ResNeXt 的各个分支的拓扑结构都是一样的，从而可减少超参的数目。分支数被命名为 cardinality，增加 cardinality 比加深和加宽（增加 filter channels）更有效。ResNeXt101_vd_64x4d 是 ResNeXt101_64x4d 的改版，区别在于 ResNeXt101_vd_64x4d 采用了 3 个 3*3 的卷积核来替代了 ResNeXt101_64x4d 中第一个 7*7 的卷积核。ResNeXt101_vd_64x4d 的 layers 为 101， 分支数(cardinality) 为 64，每个分支的输入输出 channels 为4。 该 PaddleHub Module 使用 ImageNet-2012数据集训练，接受输入图片大小为 224 x 224 x 3，支持直接通过命令行或者 Python 接口进行预测。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install resnext101_vd_64x4d_imagenet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run resnext101_vd_64x4d_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="resnext101_vd_64x4d_imagenet")
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
    $ hub install resnext101_vd_64x4d_imagenet==1.0.0
    ```
