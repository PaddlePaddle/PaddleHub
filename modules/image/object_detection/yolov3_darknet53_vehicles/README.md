# yolov3_darknet53_vehicles

|模型名称|yolov3_darknet53_vehicles|
| :--- | :---: |
|类别|图像 - 目标检测|
|网络|YOLOv3|
|数据集|百度自建大规模车辆数据集|
|是否支持Fine-tuning|否|
|模型大小|238MB|
|最新更新日期|2021-03-15|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
     <p align="center">
     <img src="https://user-images.githubusercontent.com/22424850/131529643-70ee93fc-c9f3-40df-a981-901074683beb.jpg"   width='50%' hspace='10'/>
     <br />
     </p>

- ### 模型介绍

  - 车辆检测是城市交通监控中非常重要并且具有挑战性的任务，该任务的难度在于对复杂场景中相对较小的车辆进行精准地定位和分类。该 PaddleHub Module 的网络为 YOLOv3, 其中 backbone 为 DarkNet53，采用百度自建大规模车辆数据集训练得到，支持car (汽车)、truck (卡车)、bus (公交车)、motorbike (摩托车)、tricycle (三轮车)等车型的识别。目前仅支持预测。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install yolov3_darknet53_vehicles
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run yolov3_darknet53_vehicles --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现车辆检测模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)
- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    vehicles_detector = hub.Module(name="yolov3_darknet53_vehicles")
    result = vehicles_detector.object_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = vehicles_detector.object_detection((paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def object_detection(paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='yolov3_vehicles_detect_output',
                         score_thresh=0.2,
                         visualization=True)
    ```

    - 预测API，检测输入图片中的所有车辆的位置。

    - **参数**

      - paths (list\[str\]): 图片的路径； <br/>
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式； <br/>
      - batch\_size (int): batch 的大小；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 yolov3\_vehicles\_detect\_output；<br/>
      - score\_thresh (float): 识别置信度的阈值；<br/>
      - visualization (bool): 是否将识别结果保存为图片文件。

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为:
        - data (list): 检测结果，list的每一个元素为 dict，各字段为:
          - confidence (float): 识别的置信度
          - label (str): 标签
          - left (int): 边界框的左上角x坐标
          - top (int): 边界框的左上角y坐标
          - right (int): 边界框的右下角x坐标
          - bottom (int): 边界框的右下角y坐标
        - save\_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)

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

- PaddleHub Serving可以部署一个车辆检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m yolov3_darknet53_vehicles
    ```

  - 这样就完成了一个车辆检测的服务化API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/yolov3_darknet53_vehicles"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.0.2

  修复numpy数据读取问题

* 1.0.3

  移除 fluid api

  - ```shell
    $ hub install yolov3_darknet53_vehicles==1.0.3
    ```
