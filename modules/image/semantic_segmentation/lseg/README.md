# lseg

|模型名称|lseg|
| :--- | :---: |
|类别|图像-图像分割|
|网络|LSeg|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|1.63GB|
|指标|-|
|最新更新日期|2022-09-22|


## 一、模型基本信息

- ### 应用效果展示

  - 网络结构：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/5617725d3c5640c2b24c27294437d73c83c63f78498e40b5ab2e94d01128c70c" hspace='10'/> <br />
      </p>

  - 样例结果示例：
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/2168a1e6270c40e896dfc74f2127e964ee8a8c7164aa41e3afafe1657d1e2bba" hspace='10'/>
      </p>

- ### 模型介绍

  - 文本驱动的图像语义分割模型（Language-driven Semantic Segmentation），即通过文本控制模型的分割类别实现指定类别的图像语义分割算法。



## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0  

- ### 2.安装

    - ```shell
      $ hub install lseg
      ```
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
  - ### 1、命令行预测

    ```shell
    $ hub run lseg \
        --input_path "/PATH/TO/IMAGE" \
        --labels "Category 1" "Category 2" "Category n" \
        --output_dir "lseg_output"
    ```

  - ### 2、预测代码示例

    ```python
    import paddlehub as hub
    import cv2

    module = hub.Module(name="lseg")
    result = module.segment(
        image=cv2.imread('/PATH/TO/IMAGE'),
        labels=["Category 1", "Category 2", "Category n"],
        visualization=True,
        output_dir='lseg_output'
    )
    ```

  - ### 3、API

    ```python
    def segment(
        image: Union[str, numpy.ndarray],
        labels: Union[str, List[str]],
        visualization: bool = False,
        output_dir: str = 'lseg_output'
    ) -> Dict[str, Union[numpy.ndarray, Dict[str, numpy.ndarray]]]
    ```

    - 语义分割 API

    - **参数**

      * image (Union\[str, numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      * labels (Union\[str, List\[str\]\]): 类别文本标签；
      * visualization (bool): 是否将识别结果保存为图片文件；
      * output\_dir (str): 保存处理结果的文件目录。

    - **返回**

      * res (Dict\[str, Union\[numpy.ndarray, Dict\[str, numpy.ndarray\]\]\]): 识别结果的字典，字典中包含如下元素：
        * gray (numpy.ndarray): 灰度分割结果 (GRAY)；
        * color (numpy.ndarray): 伪彩色图分割结果 (BGR)；
        * mix (numpy.ndarray): 叠加原图和伪彩色图的分割结果 (BGR)；
        * classes (Dict\[str, numpy.ndarray\]): 各个类别标签的分割抠图结果 (BGRA)。

## 四、服务部署

- PaddleHub Serving可以部署一个语义驱动的语义分割的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    ```shell
     $ hub serving start -m lseg
    ```

    - 这样就完成了一个语义驱动的语义分割服务化API的部署，默认端口号为8866。

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
        'image': cv2_to_base64(org_im),
        'labels': ["Category 1", "Category 2", "Category n"]
    }
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/lseg"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 结果转换
    results = r.json()['results']
    results = {
        'gray': base64_to_cv2(results['gray']),
        'color': base64_to_cv2(results['color']),
        'mix': base64_to_cv2(results['mix']),
        'classes': {
            k: base64_to_cv2(v) for k, v in results['classes'].items()
        }
    }

    # 保存输出
    cv2.imwrite('mix.jpg', results['mix'])
    ```

## 五、参考资料

* 论文：[Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546)

* 官方实现：[isl-org/lang-seg](https://github.com/isl-org/lang-seg)

## 六、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install lseg==1.0.0
  ```
