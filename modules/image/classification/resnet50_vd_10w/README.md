# resnet50_vd_10w

|模型名称|resnet50_vd_10w|
| :--- | :---: |
|类别|图像-图像分类|
|网络|ResNet_vd|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|92MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - ResNet系列模型是图像分类领域的重要模型之一，模型中提出的残差单元有效地解决了深度网络训练困难的问题，通过增加模型的深度提升了模型的准确率，ResNet-vd 其实就是 ResNet-D，是ResNet 原始结构的变种。该PaddleHub Module结构为ResNet_vd，使用百度自研的基于10万种类别、4千多万的有标签数据进行训练，接受输入图片大小为224 x 224 x 3，支持finetune。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install resnet50_vd_10w
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="resnet50_vd_10w")
    input_dict, output_dict, program = classifier.context(trainable=True)
    ```

- ### 2、API

  - ```python
    def context(trainable=True, pretrained=True)
    ```
    - **参数**
      - trainable (bool): 计算图的参数是否为可训练的；<br/>
      - pretrained (bool): 是否加载默认的预训练模型。

    - **返回**
      - inputs (dict): 计算图的输入，key 为 'image', value 为图片的张量；<br/>
      - outputs (dict): 计算图的输出，key 为 'classification' 和 'feature_map'，其相应的值为：
        - classification (paddle.fluid.framework.Variable): 分类结果，也就是全连接层的输出；
        - feature\_map (paddle.fluid.framework.Variable): 特征匹配，全连接层前面的那个张量。
      - context\_prog(fluid.Program): 计算图，用于迁移学习。



  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - **参数**
      - dirname: 存在模型的目录名称；<br/>
      - model_filename: 模型文件名称，默认为\_\_model\_\_; <br/>
      - params_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效); <br/>
      - combined: 是否将参数保存到统一的一个文件中。






## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install resnet50_vd_10w==1.0.0
    ```
