# unie

|模型名称|unie|
| :--- | :---: |
|类别|图像-图像编辑|
|网络|UNIE|
|数据集|LOL|
|是否支持Fine-tuning|否|
|模型大小|42.4MB|
|指标|-|
|最新更新日期|2022-10-20|


## 一、模型基本信息

- ### 应用效果展示

  - 样例结果示例：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/e6b0e10ffb85427fa6fc6efcc4006ea0a2acf70ce7de4ca1bd0fc6a40ad9a270" hspace='10'/>
      <img src="https://ai-studio-static-online.cdn.bcebos.com/edea928e735e409e89ffe22c26460568ce6602bad0ba418abdfa627a94017eba" hspace='10'/>
      </p>

- ### 模型介绍

  - UNIE（Unsupervised Night Image Enhancement），是一个基于卷积神经网络的图像夜景增强模型。UNIE 这篇论文通过引入了一种无监督的方法，整合了一个层分解网络和一个光效应抑制网络。给定一个单一的夜间图像作为输入，分解网络在无监督的特定层先验损失的指导下，学习分解阴影、反射和光效应层。光效应抑制网络进一步抑制了光效应，同时增强了黑暗区域的照明度。



## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0  

- ### 2.安装

    - ```shell
      $ hub install unie
      ```
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
  - ### 1、命令行预测

    ```shell
    $ hub run unie \
        --input_path "/PATH/TO/IMAGE" \
        --output_dir "unie_output"
    ```

  - ### 2、预测代码示例

    ```python
    import paddlehub as hub
    import cv2

    module = hub.Module(name="unie")
    result = module.night_enhancement(
        image=cv2.imread('/PATH/TO/IMAGE'),
        visualization=True,
        output_dir='unie_output'
    )
    ```

  - ### 3、API

    ```python
    def night_enhancement(
        image: Union[str, numpy.ndarray],
        visualization: bool = True,
        output_dir: str = "unie_output"
    ) -> numpy.ndarray
    ```

    - 夜景增强 API

    - **参数**

      * image (Union\[str, numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      * visualization (bool): 是否将识别结果保存为图片文件；
      * output\_dir (str): 保存处理结果的文件目录。

    - **返回**

      * res (numpy.ndarray): 图像夜景增强结果 (BGR)；

## 四、服务部署

- PaddleHub Serving 可以部署一个图像夜景增强的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    ```shell
     $ hub serving start -m unie
    ```

    - 这样就完成了一个图像夜景增强服务化API的部署，默认端口号为8866。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json
    import base64

    import cv2
    import numpy as np

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tobytes()).decode('utf8')

    def base64_to_cv2(b64str):
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.frombuffer(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

    # 发送HTTP请求
    org_im = cv2.imread('/PATH/TO/IMAGE')
    data = {
        'image': cv2_to_base64(org_im)
    }
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/unie"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 结果转换
    results = r.json()['results']
    results = base64_to_cv2(results)

    # 保存结果
    cv2.imwrite('output.jpg', results)
    ```

## 五、参考资料

* 论文：[Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression](https://arxiv.org/abs/2207.10564)

* 官方实现：[jinyeying/night-enhancement](https://github.com/jinyeying/night-enhancement)

## 六、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install unie==1.0.0
  ```
