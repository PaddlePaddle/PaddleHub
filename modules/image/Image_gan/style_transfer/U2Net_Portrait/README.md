# U2Net_Portrait

|模型名称|U2Net_Portrait|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|U^2Net|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|254MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/07f73466f3294373965e06c141c4781992f447104a94471dadfabc1c3d920861"  height='50%' hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://ai-studio-static-online.cdn.bcebos.com/c6ab02cf27414a5ba5921d9e6b079b487f6cda6026dc4d6dbca8f0167ad7cae3"   height='50%' hspace='10'/>
    <br />
    输出图像
    <br />
    </p>


- ### 模型介绍

  - U2Net_Portrait 可以用于提取人脸的素描结果。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install U2Net_Portrait
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="U2Net_Portrait")
    result = model.Portrait_GEN(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.Portrait_GEN(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def Portrait_GEN(images=None,
                    paths=None,
                    scale=1,
                    batch_size=1,
                    output_dir='output',
                    face_detection=True,
                    visualization=False):
    ```

    - 人脸画像生成API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 输入图像路径；<br/>
      - scale (float) : 缩放因子（与face_detection相关联)；<br/>
      - batch_size (int) : batch大小；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 output；<br/>
      - visualization (bool) : 是否将结果保存为图片文件；；<br/>

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**
      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]



## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install U2Net_Portrait==1.0.0
    ```
