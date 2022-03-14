# spinalnet_vgg16_gemstone

|Module Name|spinalnet_vgg16_gemstone|
| :--- | :---: |
|Category|image classification|
|Network|vgg16|
|Dataset|gemstone|
|Fine-tuning supported or not|No|
|Module Size|1.5GB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - This module is based on SpinalNet trained on gemstone dataset, and can be used to classify a gemstone.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)


- ### 2、Installation

  - ```shell
    $ hub install spinalnet_vgg16_gemstone
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run spinalnet_vgg16_gemstone --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="spinalnet_vgg16_gemstone")
    result = classifier.predict(['/PATH/TO/IMAGE'])
    print(result)
    ```

- ### 3、API

  - ```python
    def predict(images)
    ```
    - classification API.
    - **Parameters**
      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;

    - **Return**
      - result(list[dict]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability





## IV.Release Note

* 1.0.0

  First release
  - ```shell
    $ hub install spinalnet_vgg16_gemstone==1.0.0
    ```
