# cyclegan_cityscapes

|模型名称|cyclegan_cityscapes|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|CycleGAN|
|数据集|Cityscapes|
|是否支持Fine-tuning|否|
|模型大小|33MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：

    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/137839740-4be4cf40-816f-401e-a73f-6cda037041dd.png"  width = "450" height = "300" hspace='10'/>
     <br />
    输入图像
     <br />
    <img src="https://user-images.githubusercontent.com/35907364/137839777-89fc705b-f0d7-4a93-94e2-76c0d3c5a0b0.png"  width = "450" height = "300" hspace='10'/>
     <br />
    输出图像
     <br />
    </p>


- ### 模型介绍

  - CycleGAN是生成对抗网络（Generative Adversarial Networks ）的一种，与传统的GAN只能单向生成图片不同，CycleGAN可以同时完成两个domain的图片进行相互转换。该PaddleHub Module使用Cityscapes数据集训练完成，支持图片从实景图转换为语义分割结果，也支持从语义分割结果转换为实景图。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.4.0

  - paddlehub >= 1.1.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install cyclegan_cityscapes==1.0.0
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
 

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run cyclegan_cityscapes --input_path "/PATH/TO/IMAGE"
    ```
  - **参数**

    - input_path ：指定图片路径。



- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    cyclegan = hub.Module(name="cyclegan_cityscapes")

    test_img_path = "/PATH/TO/IMAGE"

    # set input dict
    input_dict = {"image": [test_img_path]}

    # execute predict and print the result
    results = cyclegan.generate(data=input_dict)
    print(results)
    ```

- ### 3、API

  - ```python
    def generate(data)
    ```

    - 风格转换API，用于图像生成。

    - **参数**

      - data： dict 类型，有以下字段:
          - image (list\[str\])： list中每个元素为待转换的图片路径。

    - **返回**
      - res (list\[str\]): 每个元素为对应输入图片的预测结果。预测结果为dict类型，有以下字段：
          - origin: 原输入图片路径.
          - generated: 生成图片的路径。



## 四、更新历史

* 1.0.0

  初始发布

