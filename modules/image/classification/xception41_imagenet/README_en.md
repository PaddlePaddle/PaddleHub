# xception41_imagenet

|Module Name|xception41_imagenet|
| :--- | :---: |
|Category|image classification|
|Network|Xception|
|Dataset|ImageNet-2012|
|Fine-tuning supported or not|No|
|Module Size|MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - Xception is a model proposed by Google in 2016, which is an improvement on Inception V3. This module is based on Xception41, trained on ImageNet-2012, and can predict an image of size 224*224*3.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)


- ### 2、Installation

  - ```shell
    $ hub install xception41_imagenet
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run xception41_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="xception41_imagenet")
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
    $ hub install xception41_imagenet==1.0.0
    ```
