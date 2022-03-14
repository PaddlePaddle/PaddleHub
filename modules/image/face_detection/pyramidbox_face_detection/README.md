# pyramidbox_face_detection

|模型名称|pyramidbox_face_detection|
| :--- | :---: |
|类别|图像 - 人脸检测|
|网络|PyramidBox|
|数据集|WIDER FACE数据集|
|是否支持Fine-tuning|否|
|模型大小|220MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/131602468-351eb3fb-81e3-4294-ac8e-b49a3a0232cb.jpg"   width='50%' hspace='10'/>
    <br />
    </p>


- ### 模型介绍

  - PyramidBox是一种基于SSD的单阶段人脸检测器，它利用上下文信息解决困难人脸的检测问题。PyramidBox在六个尺度的特征图上进行不同层级的预测。该工作主要包括以下模块：LFPN、PyramidAnchors、CPM、Data-anchor-sampling。该PaddleHub Module的预训练数据集为WIDER FACE数据集，可支持预测。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install pyramidbox_face_detection
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run pyramidbox_face_detection --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现人脸检测模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    face_detector = hub.Module(name="pyramidbox_face_detection")
    result = face_detector.face_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = face_detector.face_detection(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def face_detection(images=None,
                       paths=None,
                       use_gpu=False,
                       output_dir='detection_result',
                       visualization=False,  
                       score_thresh=0.15)
    ```

    - 检测输入图片中的所有人脸位置。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；<br/>
      - paths (list\[str\]): 图片的路径；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 detection\_result；<br/>
      - visualization (bool): 是否将识别结果保存为图片文件；<br/>
      - score_thresh (float): 置信度的阈值。

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为:
        - path (str): 原输入图片的路径
        - data (list): 检测结果，list的每一个元素为 dict，各字段为:
          - confidence (float): 识别的置信度
          - left (int): 边界框的左上角x坐标
          - top (int): 边界框的左上角y坐标
          - right (int): 边界框的右下角x坐标
          - bottom (int): 边界框的右下角y坐标


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

- PaddleHub Serving可以部署一个在线人脸检测服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m pyramidbox_face_detection
    ```

  - 这样就完成了一个人脸检测服务化API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/pyramidbox_face_detection"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.1.0

  修复numpy数据读取问题
  - ```shell
    $ hub install pyramidbox_face_detection==1.1.0
    ```
