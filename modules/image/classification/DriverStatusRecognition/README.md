# DriverStatusRecognition

|模型名称|DriverStatusRecognition|
| :--- | :---: |
|类别|图像-图像分类|
|网络|MobileNetV3_small_ssld|
|数据集|分心司机检测数据集|
|是否支持Fine-tuning|否|
|模型大小|6MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - 驾驶员状态识别（DriverStatusRecognition），该模型可挖掘出人在疲劳状态下的表情特征，然后将这些定性的表情特征进行量化，提取出面部特征点及特征指标作为判断依据，再结合实验数据总结出基于这些参数的识别方法，最后输入获取到的状态数据进行识别和判断。该PaddleHub Module支持API预测及命令行预测。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

  - paddlex >= 1.3.7


- ### 2、安装

  - ```shell
    $ hub install DriverStatusRecognition
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

- ### 3、在线体验
  [AI Studio 快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1649513)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run DriverStatusRecognition --input_path /PATH/TO/IMAGE
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

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

    - **参数**
      - images：list类型，待检测的图像。

    - **返回**
      - result：list类型，每个元素为对应输入图片的预测结果。





## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install DriverStatusRecognition==1.0.0
    ```
