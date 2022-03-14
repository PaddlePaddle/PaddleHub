# food_classification

|模型名称|food_classification|
| :--- | :---: |
|类别|图像-图像分类|
|网络|ResNet50_vd_ssld|
|数据集|美食数据集|
|是否支持Fine-tuning|否|
|模型大小|91MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - 美食分类（food_classification），该模型可识别苹果派，小排骨，烤面包，牛肉馅饼，牛肉鞑靼。该PaddleHub Module支持API预测及命令行预测。

## 二、安装

- ### 1、环境依赖  

  - paddlehub >= 2.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

  - paddlex >= 1.3.7


- ### 2、安装

  - ```shell
    $ hub install food_classification
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run food_classification --input_path /PATH/TO/IMAGE
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

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
    - 分类接口API。
    - **参数**
      - images：list类型，待检测的图像。

    - **返回**
      - result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型:
        - category_id (int): 类别的id；
        - category（str）: 类别;
        - score（float）: 准确率





## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install food_classification==1.0.0
    ```
