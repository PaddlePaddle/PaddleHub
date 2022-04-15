# food_classification

|Module Name|food_classification|
| :--- | :---: |
|Category|image classification|
|Network|ResNet50_vd_ssld|
|Dataset|Food Dataset|
|Fine-tuning supported or not|No|
|Module Size|91MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - This module can be used for food classification.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlehub >= 2.0.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

  - paddlex >= 1.3.7


- ### 2、Installation

  - ```shell
    $ hub install food_classification
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run food_classification --input_path /PATH/TO/IMAGE
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="food_classification")
    images = [cv2.imread('/PATH/TO/IMAGE')]
    results = classifier.predict(images=images)
    for result in results:
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
        - category_id (int): category id；
        - category（str）: category name;
        - score（float）: probability





## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install food_classification==1.0.0
    ```
