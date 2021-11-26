# se_resnext50_32x4d_imagenet

|Module Name|se_resnext50_32x4d_imagenet|
| :--- | :---: |
|Category|image classification|
|Network|SE_ResNeXt|
|Dataset|ImageNet-2012|
|Fine-tuning supported or not|No|
|Module Size|107MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - Squeeze-and-Excitation Networks是由Momenta在2017年提出的一种图像分类结构.该结构通过对特征通道间的相关性进行建模，把重要的特征进行强化来提升准确率.SE_ResNeXt基于ResNeXt模型添加了SE Block，并获得了2017 ILSVR竞赛的冠军.该PaddleHub Module结构为SE_ResNeXt50_32x4d，基于ImageNet-2012数据集训练，接受输入图片大小为224 x 224 x 3，支持直接通过命令行或者Python接口进行预测.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [How to install PaddleHub]()


- ### 2、Installation

  - ```shell
    $ hub install se_resnext50_32x4d_imagenet
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run se_resnext50_32x4d_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

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
    - classification API.
    - **Parameters**
      - data (dict): key is "image", value is a list of image paths

    - **Return**
      - result(list[dict]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability





## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install se_resnext50_32x4d_imagenet==1.0.0
    ```
