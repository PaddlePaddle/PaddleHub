# food_classification

|Module Name|food_classification|
| :--- | :---: |
|Category|image classification|
|Network|ResNet50_vd_ssld|
|Dataset|美食Dataset|
|Fine-tuning supported or not|No|
|Module Size|91MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - 美食分类（food_classification），该模型可识别苹果派，小排骨，烤面包，牛肉馅饼，牛肉鞑靼.该PaddleHub Module支持API预测及命令行预测.

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [How to install PaddleHub]()

  - paddlex >= 1.3.7


- ### 2、Installation

  - ```shell
    $ hub install food_classification
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run food_classification --input_path /PATH/TO/IMAGE
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

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
      - images：list类型，待检测的图像.

    - **Return**
      - result(list[dict]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability
        - category_id (int): 类别的id；
        - category（str）: 类别;
        - score（float）: 准确率





## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install food_classification==1.0.0
    ```
