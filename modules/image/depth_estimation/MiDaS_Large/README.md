# MiDaS_Large

|模型名称|MiDaS_Large|
| :--- | :---: |
|类别|图像 - 深度估计|
|网络|-|
|数据集|3D Movies, WSVD, ReDWeb, MegaDepth|
|是否支持Fine-tuning|否|
|模型大小|399MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://img-blog.csdnimg.cn/20201227112600975.jpg"  width='70%' hspace='10'/> <br />
    </p>


- ### 模型介绍

  - MiDaS_Large是一个单目深度估计模型，模型可通过输入图像估计其中的深度信息。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)  

- ### 2、安装

  - ```shell
    $ hub install MiDaS_Large
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="MiDaS_Large")
    result = model.depth_estimation(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.depth_estimation(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def depth_estimation(images=None,
                    paths=None,
                    batch_size=1,
                    output_dir='output',
                    visualization=False):
    ```

    - 深度估计API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 图片的路径；<br/>
      - batch_size (int) : batch 的大小；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 output；<br/>
      - visualization (bool) : 是否将结果保存为图片文件。

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**
      - res (list\[numpy.ndarray\]): 图像深度数据，ndarray.shape 为 \[H, W\]


## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install MiDaS_Large==1.0.0
    ```
