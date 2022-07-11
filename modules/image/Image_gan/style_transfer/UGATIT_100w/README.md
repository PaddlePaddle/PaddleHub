# UGATIT_100w

|模型名称|UGATIT_100w|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|U-GAT-IT|
|数据集|selfie2anime|
|是否支持Fine-tuning|否|
|模型大小|41MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/d130fabd8bd34e53b2f942b3766eb6bbd3c19c0676d04abfbd5cc4b83b66f8b6"  height='80%' hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://ai-studio-static-online.cdn.bcebos.com/8538af03b3f14b1884fcf4eec48965baf939e35a783d40129085102057438c77"   height='80%' hspace='10'/>
    <br />
    输出图像
    <br />
    </p>


- ### 模型介绍

  - UGATIT图像风格转换模型, 模型可将输入的人脸图像转换成动漫风格, 模型详情请参考[UGATIT-Paddle开源项目](https://github.com/miraiwk/UGATIT-paddle)。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.8.0  

  - paddlehub >= 1.8.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install UGATIT_100w
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="UGATIT_100w")
    result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def style_transfer(images=None,
                       paths=None,
                       batch_size=1,
                       output_dir='output',
                       visualization=False)
    ```

    - 风格转换API，将输入的人脸图像转换成动漫风格。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 图片的路径；<br/>
      - batch\_size (int): batch的大小；<br/>
      - visualization (bool): 是否将识别结果保存为图片文件；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 output；<br/>

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**
      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]


## 四、服务部署

- PaddleHub Serving可以部署一个在线图像风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m UGATIT_100w
    ```

  - 这样就完成了一个图像风格转换的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/UGATIT_100w"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install UGATIT_100w==1.0.0
    ```
