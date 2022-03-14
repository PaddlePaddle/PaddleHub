# photopen

|模型名称|photopen|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|SPADEGenerator|
|数据集|coco_stuff|
|是否支持Fine-tuning|否|
|模型大小|74MB|
|最新更新日期|2021-12-14|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://camo.githubusercontent.com/22e94b0c7278af08da8c475a3d968ba2f3cd565fcb2ad6b9a165c8a65f2d12f8/68747470733a2f2f61692d73747564696f2d7374617469632d6f6e6c696e652e63646e2e626365626f732e636f6d2f39343733313032336561623934623162393762396361383062643362333038333063393138636631363264303436626438383534306464613435303239356133"  width = "90%"  hspace='10'/>
    <br />

- ### 模型介绍

  - 本模块采用一个像素风格迁移网络 Pix2PixHD，能够根据输入的语义分割标签生成照片风格的图片。为了解决模型归一化层导致标签语义信息丢失的问题，向 Pix2PixHD 的生成器网络中添加了 SPADE（Spatially-Adaptive
      Normalization）空间自适应归一化模块，通过两个卷积层保留了归一化时训练的缩放与偏置参数的空间维度，以增强生成图片的质量。语义风格标签图像可以参考[coco_stuff数据集](https://github.com/nightrome/cocostuff)获取, 也可以通过[PaddleGAN repo中的该项目](https://github.com/PaddlePaddle/PaddleGAN/blob/87537ad9d4eeda17eaa5916c6a585534ab989ea8/docs/zh_CN/tutorials/photopen.md)来自定义生成图像进行体验。



## 二、安装

- ### 1、环境依赖  
  - ppgan

- ### 2、安装

  - ```shell
    $ hub install photopen
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run photopen --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像生成模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="photopen")
    input_path = ["/PATH/TO/IMAGE"]
    # Read from a file
    module.photo_transfer(paths=input_path, output_dir='./transfer_result/', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    photo_transfer(images=None, paths=None, output_dir='./transfer_result/', use_gpu=False, visualization=True):
    ```
    - 图像转换生成API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 图片的路径；<br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization(bool): 是否保存结果到本地文件夹


## 四、服务部署

- PaddleHub Serving可以部署一个在线图像转换生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m photopen
    ```

  - 这样就完成了一个图像转换生成的在线服务API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    import cv2
    import base64


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    # 发送HTTP请求
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/photopen"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install photopen==1.0.0
    ```
