# se_resnext50_32x4d_imagenet

|模型名称|se_resnext50_32x4d_imagenet|
| :--- | :---: |
|类别|图像-图像分类|
|网络|SE_ResNeXt|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|是|
|模型大小|107MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - Squeeze-and-Excitation Networks是由Momenta在2017年提出的一种图像分类结构。该结构通过对特征通道间的相关性进行建模，把重要的特征进行强化来提升准确率。SE_ResNeXt基于ResNeXt模型添加了SE Block，并获得了2017 ILSVR竞赛的冠军。该PaddleHub Module结构为SE_ResNeXt50_32x4d，基于ImageNet-2012数据集训练，接受输入图片大小为224 x 224 x 3，支持直接通过命令行或者Python接口进行预测。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install se_resnext50_32x4d_imagenet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run se_resnext50_32x4d_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="se_resnext50_32x4d_imagenet")
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
    $ hub install se_resnext50_32x4d_imagenet==1.0.0
    ```
