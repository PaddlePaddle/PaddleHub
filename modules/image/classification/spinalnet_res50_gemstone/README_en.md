# spinalnet_res50_gemstone

|Module Name|spinalnet_res50_gemstone|
| :--- | :---: |
|Category|image classification|
|Network|resnet50|
|Dataset|gemstone|
|Fine-tuning supported or not|No|
|Module Size|137MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - 使用PaddleHub的SpinalNet预训练模型进行宝石识别或finetune并完成宝石的预测任务.
## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [How to install PaddleHub]()


- ### 2、Installation

  - ```shell
    $ hub install spinalnet_res50_gemstone
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run spinalnet_res50_gemstone --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="spinalnet_res50_gemstone")
    result = classifier.predict(['/PATH/TO/IMAGE'])
    print(result)
    ```

- ### 3、API

  - ```python
    def predict(images)
    ```
    - classification API.
    - **Parameters**
      - images: list类型，待预测的图像.

    - **Return**
      - result(list[dict]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability





## IV.Release Note

* 1.0.0

  First release
  - ```shell
    $ hub install spinalnet_res50_gemstone==1.0.0
    ```
