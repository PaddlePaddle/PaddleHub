# resnet_v2_50_imagenet

|Module Name|resnet_v2_50_imagenet|
| :--- | :---: |
|Category |Image classification|
|Network|ResNet V2|
|Dataset|ImageNet-2012|
|Fine-tuning supported or not|No|
|Module Size|99MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I. Basic Information

- ### Application Effect Display

  - This module utilizes ResNet50 structure and it is trained on ImageNet-2012.

## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)


- ### 2、Installation

  - ```shell
    $ hub install resnet_v2_50_imagenet
    ```
  - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run resnet_v2_50_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="resnet_v2_50_imagenet")
    test_img_path = "/PATH/TO/IMAGE"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    ```

- ### 3、API

  - ```python
    def classification(data)
    ```
    - Prediction API for classification.

    - **Parameter**
      - data (dict): Key is 'image'，value is the list of image path.

    - **Return**
      - result (list[dict]): The list of classification results，key is the prediction label, value is the corresponding confidence.





## IV. Release Note

- 1.0.0

  First release

- 1.0.1

  Fix encoding problem in python2

  - ```shell
    $ hub install resnet_v2_50_imagenet==1.0.1
    ```
