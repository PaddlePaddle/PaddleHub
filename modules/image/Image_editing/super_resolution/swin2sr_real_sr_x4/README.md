# swin2sr_real_sr_x4

|模型名称|swin2sr_real_sr_x4|
| :--- | :---: |
|类别|图像-图像编辑|
|网络|Swin2SR|
|数据集|DIV2K / Flickr2K|
|是否支持Fine-tuning|否|
|模型大小|68.4MB|
|指标|-|
|最新更新日期|2022-10-25|


## 一、模型基本信息

- ### 应用效果展示

  - 网络结构：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/884d4d4472b44bf1879606374ed64a7e8d2fec0bcf034285a5cecfc582e8cd65" hspace='10'/> <br />
      </p>

  - 样例结果示例：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/c5517af6c3f944c4b281aedc417a4f8c02c0a969d0dd494c9106c4ff2709fc2f" hspace='10'/>
      <img src="https://ai-studio-static-online.cdn.bcebos.com/183c5821029f45bbb78d1700ab8297baabba15f82ab4467e88414bbed056ccf0" hspace='10'/>
      </p>

- ### 模型介绍

  - Swin2SR 是一个基于 Swin Transformer v2 的图像超分辨率模型。swin2sr_real_sr_x4 是基于 Swin2SR 的 4 倍现实图像超分辨率模型。



## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0  

- ### 2.安装

    - ```shell
      $ hub install swin2sr_real_sr_x4
      ```
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
  - ### 1、命令行预测

    ```shell
    $ hub run swin2sr_real_sr_x4 \
        --input_path "/PATH/TO/IMAGE" \
        --output_dir "swin2sr_real_sr_x4_output"
    ```

  - ### 2、预测代码示例

    ```python
    import paddlehub as hub
    import cv2

    module = hub.Module(name="swin2sr_real_sr_x4")
    result = module.real_sr(
        image=cv2.imread('/PATH/TO/IMAGE'),
        visualization=True,
        output_dir='swin2sr_real_sr_x4_output'
    )
    ```

  - ### 3、API

    ```python
    def real_sr(
        image: Union[str, numpy.ndarray],
        visualization: bool = True,
        output_dir: str = "swin2sr_real_sr_x4_output"
    ) -> numpy.ndarray
    ```

    - 超分辨率 API

    - **参数**

      * image (Union\[str, numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      * visualization (bool): 是否将识别结果保存为图片文件；
      * output\_dir (str): 保存处理结果的文件目录。

    - **返回**

      * res (numpy.ndarray): 图像超分辨率结果 (BGR)；

## 四、服务部署

- PaddleHub Serving 可以部署一个图像超分辨率的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    ```shell
     $ hub serving start -m swin2sr_real_sr_x4
    ```

    - 这样就完成了一个图像超分辨率服务化API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/swin2sr_real_sr_x4"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 结果转换
    results = r.json()['results']
    results = base64_to_cv2(results)

    # 保存结果
    cv2.imwrite('output.jpg', results)
    ```

## 五、参考资料

* 论文：[Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345)

* 官方实现：[mv-lab/swin2sr](https://github.com/mv-lab/swin2sr/)

## 六、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install swin2sr_real_sr_x4==1.0.0
  ```
