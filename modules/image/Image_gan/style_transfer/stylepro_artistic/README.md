# stylepro_artistic

|模型名称|stylepro_artistic|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|StyleProNet|
|数据集|MS-COCO + WikiArt|
|是否支持Fine-tuning|否|
|模型大小|28MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://paddlehub.bj.bcebos.com/resources/style.png"  width='80%' hspace='10'/> <br />
    </p>

- ### 模型介绍

  - 艺术风格迁移模型可以将给定的图像转换为任意的艺术风格。本模型StyleProNet整体采用全卷积神经网络架构(FCNs)，通过encoder-decoder重建艺术风格图片。StyleProNet的核心是无参数化的内容-风格融合算法Style Projection，模型规模小，响应速度快。模型训练的损失函数包含style loss、content perceptual loss以及content KL loss，确保模型高保真还原内容图片的语义细节信息与风格图片的风格信息。预训练数据集采用MS-COCO数据集作为内容端图像，WikiArt数据集作为风格端图像。更多详情请参考StyleProNet论文[https://arxiv.org/abs/2003.07694](https://arxiv.org/abs/2003.07694)。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0     | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)  

- ### 2、安装

  - ```shell
    $ hub install stylepro_artistic
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run stylepro_artistic --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现风格转换模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)
- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    stylepro_artistic = hub.Module(name="stylepro_artistic")
    result = stylepro_artistic.style_transfer(
    images=[{
        'content': cv2.imread('/PATH/TO/CONTENT_IMAGE'),
        'styles': [cv2.imread('/PATH/TO/STYLE_IMAGE')]
    }])

    # or
    # result = stylepro_artistic.style_transfer(
    #     paths=[{
    #         'content': '/PATH/TO/CONTENT_IMAGE',
    #         'styles': ['/PATH/TO/STYLE_IMAGE']
    #     }])
    ```

- ### 3、API

  - ```python
    def style_transfer(images=None,
                       paths=None,
                       alpha=1,
                       use_gpu=False,
                       visualization=False,
                       output_dir='transfer_result')
    ```

    - 对图片进行风格转换。

    - **参数**
      - images (list\[dict\]): ndarray 格式的图片数据。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
        - content (numpy.ndarray): 待转换的图片，shape 为 \[H, W, C\]，BGR格式；<br/>
        - styles (list\[numpy.ndarray\]) : 作为底色的风格图片组成的列表，各个图片数组的shape 都是 \[H, W, C\]，BGR格式；<br/>
        - weights (list\[float\], optioal) : 各个 style 对应的权重。当不设置 weights 时，默认各个 style 有着相同的权重；<br/>
      - paths (list\[str\]): 图片的路径。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
        - content (str): 待转换的图片的路径；<br/>
        - styles (list\[str\]) : 作为底色的风格图片的路径；<br/>
        - weights (list\[float\], optioal) : 各个 style 对应的权重。当不设置 weights 时，各个 style 的权重相同；<br/>
      - alpha (float) : 转换的强度，\[0, 1\] 之间，默认值为1；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization (bool): 是否将结果保存为图片，默认为 False; <br/>
      - output\_dir (str): 图片的保存路径，默认设为 transfer\_result。

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 OrderedDict，关键字有 date, save\_path，相应取值为：
        - save\_path (str): 保存图片的路径
        - data (numpy.ndarray): 风格转换的图片数据


  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - 将模型保存到指定路径。

    - **参数**

      - dirname: 存在模型的目录名称； <br/>
      - model\_filename: 模型文件名称，默认为\_\_model\_\_； <br/>
      - params\_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)；<br/>
      - combined: 是否将参数保存到统一的一个文件中。


## 四、服务部署

- PaddleHub Serving可以部署一个在线风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m stylepro_artistic
    ```

  - 这样就完成了一个风格转换服务化API的部署，默认端口号为8866。

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
    data = {'images':[
    {
        'content':cv2_to_base64(cv2.imread('/PATH/TO/CONTENT_IMAGE')),
        'styles':[cv2_to_base64(cv2.imread('/PATH/TO/STYLE_IMAGE'))]
    }
    ]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stylepro_artistic"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(base64_to_cv2(r.json()["results"][0]['data']))
    ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.0.3

  移除 fluid api

  - ```shell
    $ hub install stylepro_artistic==1.0.3
    ```
