# DriverStatusRecognition

|Module Name|DriverStatusRecognition|
| :--- | :---: |
|Category|image classification|
|Network|MobileNetV3_small_ssld|
|Dataset|分心司机检测Dataset|
|Fine-tuning supported or not|No|
|Module Size|6MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - 驾驶员状态识别（DriverStatusRecognition），该模型可挖掘出人在疲劳状态下的表情特征，然后将这些定性的表情特征进行量化，提取出面部特征点及特征指标作为判断依据，再结合实验数据总结出基于这些Parameters的识别方法，最后输入获取到的状态数据进行识别和判断.该PaddleHub Module支持API预测及命令行预测.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [How to install PaddleHub]()

  - paddlex >= 1.3.7


- ### 2、Installation

  - ```shell
    $ hub install DriverStatusRecognition
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

- ### 3、在线体验
  [AI Studio 快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1649513)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run DriverStatusRecognition --input_path /PATH/TO/IMAGE
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="DriverStatusRecognition")
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
    $ hub install DriverStatusRecognition==1.0.0
    ```
