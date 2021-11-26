# marine_biometrics

|Module Name|marine_biometrics|
| :--- | :---: |
|Category|image classification|
|Network|ResNet50_vd_ssld|
|Dataset|Fish4Knowledge|
|Fine-tuning supported or not|No|
|Module Size|84MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - 海洋生物识别（marine_biometrics），该模型可准确识别鱼的种类.该PaddleHub Module支持API预测及命令行预测.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [How to install PaddleHub]()


- ### 2、Installation

  - ```shell
    $ hub install marine_biometrics
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run marine_biometrics --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="marine_biometrics")
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
      - images：list类型，待检测的图像.

    - **Return**
      - result(list[dict]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability





## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install marine_biometrics==1.0.0
    ```
