# Extract_Line_Draft

|模型名称|Extract_Line_Draft|
| :--- | :---: | 
|类别|图像-图像分割|
|网络|-|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|259MB|
|指标|-|
|最新更新日期|2021-02-26|


## 一、模型基本信息

- ### 应用效果展示

  - 样例结果示例：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/1c30757e069541a18dc89b92f0750983b77ad762560849afa0170046672e57a3" width = "337" height = "505" hspace='10'/> <img src="https://ai-studio-static-online.cdn.bcebos.com/7ef00637e5974be2847317053f8abe97236cec75fba14f77be2c095529a1eeb3" width = "337" height = "505" hspace='10'/>
      </p>

- ### 模型介绍

  - 提取线稿（Extract_Line_Draft），该模型可自动根据彩色图生成线稿图。该PaddleHub Module支持API预测及命令行预测。


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0  

- ### 2.安装

    - ```shell
      $ hub install Extract_Line_Draft
      ```
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
  - ### 1、命令行预测

    ```shell
    $ hub run Extract_Line_Draft --input_path "testImage" --use_gpu True
    ```

  - ### 2、预测代码示例

    ```python
    import paddlehub as hub

    Extract_Line_Draft_test = hub.Module(name="Extract_Line_Draft")

    test_img = "testImage.png"

    # execute predict
    Extract_Line_Draft_test.ExtractLine(test_img, use_gpu=True)
    ```
  
  - ### 3、API

    ```python
    def ExtractLine(image, use_gpu=False)
    ```

    - 预测API，用于图像分割得到人体解析。

    - **参数**

      * image(str): 待检测的图片路径
      * use_gpu (bool): 是否使用 GPU


## 四、更新历史

* 1.0.0

  初始发布

* 1.1.0

  移除 Fluid API

  ```shell
  $ hub install Extract_Line_Draft == 1.1.0
  ```