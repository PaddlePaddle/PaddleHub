# chinese_text_detection_db_server

|模型名称|chinese_text_detection_db_server|
| :--- | :---: |
|类别|图像-文字识别|
|网络|Differentiable Binarization|
|数据集|icdar2015数据集|
|是否支持Fine-tuning|否|
|模型大小|47MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133101419-2d175dc4-2274-404d-b9f3-802f398a6ce4.jpg" width="500" alt="package" >
</p>

- ### 模型介绍

  - DB（Differentiable Binarization）是一种基于分割的文本检测算法。此类算法可以更好地处理弯曲等不规则形状文本，因此检测效果往往会更好。但其后处理步骤中将分割结果转化为检测框的流程复杂，耗时严重。DB将二值化阈值加入训练中学习，可以获得更准确的检测边界，从而简化后处理流程。该Module是一个超轻量级文本检测模型，支持直接预测。

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133101635-7fb142d3-9056-44da-8201-d931727d3977.png" width="800" hspace='10'/> <br />
</p>

  - 更多详情参考：[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.7.2  

  - paddlehub >= 1.6.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

  - shapely

  - pyclipper

  - ```shell
    $ pip install shapely pyclipper
    ```
  - **该Module依赖于第三方库shapely和pyclipper，使用该Module之前，请先安装shapely和pyclipper。**

- ### 2、安装

  - ```shell
    $ hub install chinese_text_detection_db_server
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run chinese_text_detection_db_server --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    text_detector = hub.Module(name="chinese_text_detection_db_server"), enable_mkldnn=True)       # mkldnn加速仅在CPU下有效
    result = text_detector.detect_text(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result =text_detector.detect_text(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    __init__(enable_mkldnn=False)
    ```
    - 构造ChineseTextDetectionDBServer对象

    - **参数**

      - enable_mkldnn(bool): 是否开启mkldnn加速CPU计算。该参数仅在CPU运行下设置有效。默认为False。

  - ```python
    def detect_text(paths=[],
                    images=[],
                    use_gpu=False,
                    output_dir='detection_result',
                    box_thresh=0.5,
                    visualization=False)
    ```
    - 预测API，检测输入图片中的所有中文文本的位置。

    - **参数**

      - paths (list\[str\]): 图片的路径；
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      - use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**
      - box\_thresh (float): 检测文本框置信度的阈值；
      - visualization (bool): 是否将识别结果保存为图片文件；
      - output\_dir (str): 图片的保存路径，默认设为 detection\_result；

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
        - data (list): 检测文本框结果，文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
        - ave_path (str): 识别结果的保存路径, 如不保存图片则save_path为''


## 四、服务部署

- PaddleHub Serving 可以部署一个目标检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m chinese_text_detection_db_server
    ```

  - 这样就完成了一个目标检测的服务化API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/chinese_text_detection_db_server"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  支持mkldnn加速CPU计算

* 1.0.2

  增加更多预训练数据，更新预训练参数

* 1.0.3

  移除 fluid api

  - ```shell
    $ hub install chinese_text_detection_db_server==1.0.3
    ```
