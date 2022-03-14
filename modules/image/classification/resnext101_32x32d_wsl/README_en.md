# resnext101_32x32d_wsl

|Module Name|resnext101_32x32d_wsl|
| :--- | :---: |
|Category|image classification|
|Network|ResNeXt_wsl|
|Dataset|ImageNet-2012|
|Fine-tuning supported or not|No|
|Module Size|1.8GB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - The scale of dataset annotated by people is close to limit, researchers in Facebook adopt a new method of transfer learning to train the network. They use hashtag to annotate images, and trained on billions of social images, then transfer to weakly supervised learning. The top-1 accuracy of ResNeXt101_32x32d_wsl on ImageNet reaches 84.97%. This module is based on ResNeXt101_32x32d_wsl, and can predict an image of size 224*224*3.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.0  

  - paddlehub >= 1.0.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)


- ### 2、Installation

  - ```shell
    $ hub install resnext101_32x32d_wsl
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run resnext101_32x32d_wsl --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="resnext101_32x32d_wsl")
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
    $ hub install resnext101_32x32d_wsl==1.0.0
    ```
