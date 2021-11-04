# seeinthedark

|模型名称|seeinthedark|
| :--- | :---: |
|类别|图像 - 暗光增强|
|网络|ConvNet|
|数据集|SID dataset|
|是否支持Fine-tuning|否|
|模型大小|120MB|
|最新更新日期|2021-11-02|
|数据指标|-|


## 一、模型基本信息  

- ### 模型介绍

  - 通过大量暗光条件下短曝光和长曝光组成的图像对，以RAW图像为输入，RGB图像为参照进行训练，该模型实现端到端直接将暗光下的RAW图像处理得到可见的RGB图像。

  - 更多详情参考：[Learning to See in the Dark](http://cchen156.github.io/paper/18CVPR_SID.pdf)



## 二、安装

- ### 1、环境依赖  
  - rawpy
  - pillow

- ### 2、安装

  - ```shell
    $ hub install seeinthedark
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
  - 在windows下安装，由于paddledet package会依赖cython-bbox以及pycocotools, 这两个包需要windows用户提前装好，可参考[cython-bbox安装](https://blog.csdn.net/qq_24739717/article/details/105588729)和[pycocotools安装](https://github.com/PaddlePaddle/PaddleX/blob/release/1.3/docs/install.md#pycocotools安装问题)
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a raw(Sony, .ARW) file
    $ hub run seeinthedark --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现暗光增强模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub

    denoiser = hub.Module(name="seeinthedark")
    input_path = "/PATH/TO/IMAGE"
    # Read from a raw file
    denoiser.denoising(input_path, output_path='./denoising_result.png', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    def denoising(input_path, output_path='./denoising_result.png', use_gpu=False)
    ```
    - 暗光增强API，完成对暗光RAW图像的降噪并处理生成RGB图像。

    - **参数**

      - input\_path (str): 暗光图像文件的路径，Sony的RAW格式; <br/>
      - output\_path (str): 结果保存的路径, 需要指定输出文件名； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>




## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install seeinthedark==1.0.0
    ```
