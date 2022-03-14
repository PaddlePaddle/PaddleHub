# Photo2Cartoon

|模型名称|Photo2Cartoon|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|U-GAT-IT|
|数据集|cartoon_data|
|是否支持Fine-tuning|否|
|模型大小|205MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://img-blog.csdnimg.cn/20201224164040624.jpg"   hspace='10'/> <br />
    </p>



- ### 模型介绍

  - 本模型封装自[小视科技photo2cartoon项目的paddlepaddle版本](https://github.com/minivision-ai/photo2cartoon-paddle)。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)  

- ### 2、安装

  - ```shell
    $ hub install Photo2Cartoon
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="Photo2Cartoon")
    result = model.Cartoon_GEN(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.Cartoon_GEN(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def Cartoon_GEN(images=None,
                    paths=None,
                    batch_size=1,
                    output_dir='output',
                    visualization=False,
                    use_gpu=False):
    ```

    - 人像卡通化图像生成API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 输入图像路径；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 output；<br/>
      - batch_size (int) : batch大小；<br/>  
      - visualization (bool) : 是否将结果保存为图片文件；；<br/>
      - use_gpu (bool) : 是否使用 GPU 进行推理。

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**
      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]



## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install Photo2Cartoon==1.0.0
    ```
