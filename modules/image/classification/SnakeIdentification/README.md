# SnakeIdentification

|模型名称|SnakeIdentification|
| :--- | :---: |
|类别|图像-图像分类|
|网络|ResNet50_vd_ssld|
|数据集|蛇种数据集|
|是否支持Fine-tuning|否|
|模型大小|84MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - 蛇种识别（SnakeIdentification），该模型可准确识别蛇的种类，并精准判断蛇的毒性。该PaddleHub Module支持API预测及命令行预测。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

  - paddlex >= 1.3.7


- ### 2、安装

  - ```shell
    $ hub install SnakeIdentification
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

- ### 3、在线体验
  [AI Studio 快速体验](https://aistudio.baidu.com/aistudio/projectdetail/1646951)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run SnakeIdentification --input_path /PATH/TO/IMAGE
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="SnakeIdentification")
    images = [cv2.imread('/PATH/TO/IMAGE')]
    results = classifier.predict(images=images)
    for result in results:
        print(result)
    ```

- ### 3、API

  - ```python
    def predict(images)
    ```
    - 分类接口API。
    - **参数**
      - images：list类型，待检测的图像。

    - **返回**
      - result：list类型，每个元素为对应输入图片的预测结果。





## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install SnakeIdentification==1.0.0
    ```
