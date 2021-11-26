# mobilenet_v2_imagenet

|Module Name|mobilenet_v2_imagenet|
| :--- | :---: |
|Category|image classification|
|Network|Mobilenet_v2|
|Dataset|ImageNet-2012|
|Fine-tuning supported or not|No|
|Module Size|15MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - MobileNet V2是Mark Sandler, Andrew Howard等人在2018年提出的一个图像分类模型，该系列模型（MobileNet）是为移动和嵌入式设备提出的高效模型，在模型Parameters较少的情况下仍然保持了较高的分类准确率.该PaddleHub Module基于ImageNet-2012数据集训练，接受输入图片大小为224 x 224 x 3，支持直接通过命令行或者Python接口进行预测.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [How to install PaddleHub]()


- ### 2、Installation

  - ```shell
    $ hub install mobilenet_v2_imagenet
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run mobilenet_v2_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="mobilenet_v2_imagenet")
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

* 1.0.1

  Fix the problem of encoding in python2

  - ```shell
    $ hub install mobilenet_v2_imagenet==1.0.1
    ```
