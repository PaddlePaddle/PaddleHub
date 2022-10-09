# fbcnn_gray

|模型名称|fbcnn_gray|
| :--- | :---: |
|类别|图像-图像编辑|
|网络|FBCNN|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|288MB|
|指标|-|
|最新更新日期|2022-10-08|


## 一、模型基本信息

- ### 应用效果展示

  - 网络结构：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/08afa15df2e54adeb39587cd7aaa9b60fc82d349bda34f51993d6304776fd374" hspace='10'/> <br />
      </p>

  - 样例结果示例：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/4804ea3fff524c578014ec98e5222b6310a4cdf1ba41448c94829399e82880b6" hspace='10'/>
      </p>

- ### 模型介绍

  - FBCNN 是一个基于卷积神经网络的 JPEG 图像伪影去除模型，它可以预测可调整的质量因子，以控制伪影重新移动和细节保留之间的权衡。



## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0  

- ### 2.安装

    - ```shell
      $ hub install fbcnn_gray
      ```
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
  - ### 1、命令行预测

    ```shell
    $ hub run fbcnn_gray \
        --input_path "/PATH/TO/IMAGE" \
        --quality_factor -1 \
        --output_dir "fbcnn_gray_output"
    ```

  - ### 2、预测代码示例

    ```python
    import paddlehub as hub
    import cv2

    module = hub.Module(name="fbcnn_gray")
    result = module.artifacts_removal(
        image=cv2.imread('/PATH/TO/IMAGE'),
        quality_factor=None,
        visualization=True,
        output_dir='fbcnn_gray_output'
    )
    ```

  - ### 3、API

    ```python
    def artifacts_removal(
        image: Union[str, numpy.ndarray],
        quality_factor: float = None,
        visualization: bool = True,
        output_dir: str = "fbcnn_gray_output"
    ) -> numpy.ndarray
    ```

    - 伪影去除 API

    - **参数**

      * image (Union\[str, numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W\]，GRAY格式；
      * quality_factor (float): 自定义质量因子（0.0 - 1.0），默认 None 为自适应；
      * visualization (bool): 是否将识别结果保存为图片文件；
      * output\_dir (str): 保存处理结果的文件目录。

    - **返回**

      * res (numpy.ndarray): 图像伪影去除结果 (GRAY)；

## 四、服务部署

- PaddleHub Serving 可以部署一个图像伪影去除的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    ```shell
     $ hub serving start -m fbcnn_gray
    ```

    - 这样就完成了一个图像伪影去除服务化API的部署，默认端口号为8866。

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
        data = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return data

    # 发送HTTP请求
    org_im = cv2.imread('/PATH/TO/IMAGE')
    data = {
        'image': cv2_to_base64(org_im)
    }
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/fbcnn_gray"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 结果转换
    results = r.json()['results']
    results = base64_to_cv2(results)

    # 保存结果
    cv2.imwrite('output.jpg', results)
    ```

## 五、参考资料

* 论文：[Towards Flexible Blind JPEG Artifacts Removal](https://arxiv.org/abs/2109.14573)

* 官方实现：[jiaxi-jiang/FBCNN](https://github.com/jiaxi-jiang/FBCNN)

## 六、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install fbcnn_gray==1.0.0
  ```
