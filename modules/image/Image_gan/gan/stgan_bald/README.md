# stgan_bald

|模型名称|stgan_bald|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|STGAN|
|数据集|CelebA|
|是否支持Fine-tuning|否|
|模型大小|287MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 详情请查看此链接：https://aistudio.baidu.com/aistudio/projectdetail/1145381

- ### 模型介绍

  - stgan_bald 以STGAN 为模型，使用 CelebA 数据集训练完成，该模型可自动根据图像生成1年、3年、5年的秃头效果。


## 二、安装

- ### 1、环境依赖  

  - paddlehub >= 1.8.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install stgan_bald
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    stgan_bald = hub.Module(name="stgan_bald")
    result = stgan_bald.bald(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = stgan_bald.bald(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def bald(images=None,
             paths=None,
             use_gpu=False,
             visualization=False,
             output_dir="bald_output")
    ```

    - 秃头生成器API预测接口, 预测输入一张人像，输出三张秃头效果(1年、3年、5年)。

    - **参数**
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 图片的路径；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization (bool): 是否将结果保存为图片，默认为 False; <br/>
      - output\_dir (str): 图片的保存路径，默认设为bald\_output。

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**

      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]

## 四、服务部署

- PaddleHub Serving可以部署一个秃头生成器服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m stgan_bald
    ```

  - 这样就完成了一个秃头生成器API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    import cv2
    import base64
    import numpy as np


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    def base64_to_cv2(b64str):
      data = base64.b64decode(b64str.encode('utf8'))
      data = np.fromstring(data, np.uint8)
      data = cv2.imdecode(data, cv2.IMREAD_COLOR)
      return data

    # 发送HTTP请求
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stgan_bald"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 保存图片 1年 3年 5年
    one_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_0']), cv2.COLOR_RGB2BGR)
    three_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_1']), cv2.COLOR_RGB2BGR)
    five_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_2']), cv2.COLOR_RGB2BGR)
    cv2.imwrite("stgan_bald_server.png", one_year)
    ```


## 五、更新历史

* 1.0.0

  初始发布
  - ```shell
    $ hub install stgan_bald==1.0.0
    ```
