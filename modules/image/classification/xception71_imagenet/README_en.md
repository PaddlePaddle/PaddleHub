# xception71_imagenet

|Module Name|xception71_imagenet|
| :--- | :---: |
|Category|image classification|
|Network|Xception|
|Dataset|ImageNet-2012|
|Fine-tuning supported or not|No|
|Module Size|147MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - Xception 全称为 Extreme Inception，是 Google 于 2016年提出的 Inception V3 的改进模型.Xception 采用了深度可分离卷积(depthwise separable convolution) 来替换原来 Inception V3 中的卷积操作，整体的网络结构是带有残差连接的深度可分离卷积层的线性堆叠.该PaddleHub Module结构为Xception71，基于ImageNet-2012数据集训练，接受输入图片大小为224 x 224 x 3，支持直接通过命令行或者 Python 接口进行预测.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [How to install PaddleHub]()


- ### 2、Installation

  - ```shell
    $ hub install xception71_imagenet
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run xception71_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="xception71_imagenet")
    test_img_path = "/PATH/TO/IMAGE"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    ```

- ### 3、API

  - ```python
    def classification(data)
    ```
    - classification API.
    - **Parameters**
      - data (dict): key is "image", value is a list of image paths

    - **Return**
      - result(list[dict]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability





## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install xception71_imagenet==1.0.0
    ```
