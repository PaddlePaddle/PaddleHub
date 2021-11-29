# resnet50_vd_10w

|Module Name|resnet50_vd_10w|
| :--- | :---: |
|Category|image classification|
|Network|ResNet_vd|
|Dataset|Baidu Dataset|
|Fine-tuning supported or not|No|
|Module Size|92MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - ResNet proposed a residual unit to solve the problem of training an extremely deep network, and improved the prediction accuracy of models. ResNet-vd is a variant of ResNet. This module is based on ResNet_vd, trained on Baidu dataset(consists of 100 thousand classes, 40 million pairs of data), and can predict an image of size 224*224*3.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub]()


- ### 2、Installation

  - ```shell
    $ hub install resnet50_vd_10w
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、预测Prediction Code Example

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
    - **Parameters**
      - trainable (bool): 计算图的Parameters是否为可训练的；<br/>
      - pretrained (bool): 是否加载默认的预训练模型.

    - **Return**
      - inputs (dict): 计算图的输入，key 为 'image', value 为图片的张量；<br/>
      - outputs (dict): 计算图的输出，key 为 'classification' 和 'feature_map'，其相应的值为：
        - classification (paddle.fluid.framework.Variable): 分类结果，也就是全连接层的输出；
        - feature\_map (paddle.fluid.framework.Variable): 特征匹配，全连接层前面的那个张量.
      - context\_prog(fluid.Program): 计算图，用于迁移学习.



  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - **Parameters**
      - dirname: 存在模型的目录名称；<br/>
      - model_filename: 模型文件名称，默认为\_\_model\_\_; <br/>
      - params_filename: Parameters文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效); <br/>
      - combined: 是否将Parameters保存到统一的一个文件中.






## V.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install resnet50_vd_10w==1.0.0
    ```
