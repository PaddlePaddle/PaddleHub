# EnlightenGAN

|模型名称|EnlightenGAN|
| :--- | :---: |
|类别|图像 - 暗光增强|
|网络|EnlightenGAN|
|数据集||
|是否支持Fine-tuning|否|
|模型大小|83MB|
|最新更新日期|2021-11-04|
|数据指标|-|


## 一、模型基本信息  

- ### 模型介绍

  - EnlightenGAN使用非成对的数据进行训练，通过设计自特征保留损失函数和自约束注意力机制，训练的网络可以应用到多种场景下的暗光增强中。

  - 更多详情参考：[EnlightenGAN: Deep Light Enhancement without Paired Supervision](https://arxiv.org/abs/1906.06972)



## 二、安装

- ### 1、环境依赖  
  - onnxruntime
  - x2paddle
  - pillow

- ### 2、安装

  - ```shell
    $ hub install EnlightenGAN
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
  - 在windows下安装，由于paddledet package会依赖cython-bbox以及pycocotools, 这两个包需要windows用户提前装好，可参考[cython-bbox安装](https://blog.csdn.net/qq_24739717/article/details/105588729)和[pycocotools安装](https://github.com/PaddlePaddle/PaddleX/blob/release/1.3/docs/install.md#pycocotools安装问题)
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run EnlightenGAN --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现暗光增强模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub

    enlightener = hub.Module(name="EnlightenGAN")
    input_path = "/PATH/TO/IMAGE"
    # Read from a raw file
    enlightener.enlightening(input_path, output_path='./enlightening_result.png', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    def enlightening(input_path, output_path='./enlightening_result.png', use_gpu=False)
    ```
    - 暗光增强API。

    - **参数**

      - input\_path (str): 输入图像文件的路径; <br/>
      - output\_path (str): 结果保存的路径, 需要指定输出文件名； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>




## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install EnlightenGAN==1.0.0
    ```
