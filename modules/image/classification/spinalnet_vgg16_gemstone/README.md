# spinalnet_vgg16_gemstone

|模型名称|spinalnet_vgg16_gemstone|
| :--- | :---: |
|类别|图像-图像分类|
|网络|vgg16|
|数据集|gemstone|
|是否支持Fine-tuning|否|
|模型大小|1.5GB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - 使用PaddleHub的SpinalNet预训练模型进行宝石识别或finetune并完成宝石的预测任务。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install spinalnet_vgg16_gemstone
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run spinalnet_vgg16_gemstone --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

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
    - 分类接口API。
    - **参数**
      - images: list类型，待预测的图像。

    - **返回**
      - result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label对应的概率





## 四、更新历史

* 1.0.0

  初始发布
  - ```shell
    $ hub install spinalnet_vgg16_gemstone==1.0.0
    ```
