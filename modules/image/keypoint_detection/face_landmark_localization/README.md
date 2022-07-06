# face_landmark_localization

| 模型名称            | face_landmark_localization |
| :------------------ | :------------------------: |
| 类别                |      图像-关键点检测       |
| 网络                |       Face_Landmark        |
| 数据集              |          AFW/AFLW          |
| 是否支持Fine-tuning |             否             |
| 模型大小            |             3M             |
| 最新更新日期        |         2021-02-26         |
| 数据指标            |             -              |

## 一、模型基本信息

- ### 应用效果展示
  - 人脸关键点（左）、模型检测效果（右）

    <p align="center">
    <img src="https://user-images.githubusercontent.com/76040149/133222449-a1c2d444-a839-4c0e-9203-30d67eeb2246.jpeg"  hspace="5" width="300"/> <img src="https://user-images.githubusercontent.com/76040149/133229934-e7357767-28e0-4253-bf71-f948de9966f1.jpg"  hspace="5" height="300"/>
    </p>

- ### 模型介绍

  - 人脸关键点检测是人脸识别和分析领域中的关键一步，它是诸如自动人脸识别、表情分析、三维人脸重建及三维动画等其它人脸相关问题的前提和突破口。该 PaddleHub Module 的模型转换自 https://github.com/lsy17096535/face-landmark ，支持同一张图中的多个人脸检测。


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 1.6.2

  - paddlehub >= 1.6.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install face_landmark_localization
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
   | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run face_landmark_localization --input_path "/PATH/TO/IMAGE"
    ```

  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    face_landmark = hub.Module(name="face_landmark_localization")

    # Replace face detection module to speed up predictions but reduce performance
    # face_landmark.set_face_detector_module(hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320"))

    result = face_landmark.keypoint_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = face_landmark.keypoint_detection(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def __init__(face_detector_module=None):
    ```

    - **参数**
      - face_detector_module (class): 人脸检测模型，默认为 ultra_light_fast_generic_face_detector_1mb_640.

  - ```python
    def keypoint_detection(images=None,
                           paths=None,
                           batch_size=1,
                           use_gpu=False,
                           output_dir='face_landmark_output',
                           visualization=False):
    ```

    - 识别输入图片中的所有人脸关键点，每张人脸检测出68个关键点（人脸轮廓17个点，左右眉毛各5个点，左右眼睛各6个点，鼻子9个点，嘴巴20个点）


    - **参数**

      - images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
      - paths (list[str]): 图片的路径；
      - batch_size (int): batch 的大小；
      - use_gpu (bool): 是否使用 GPU；
      - visualization (bool): 是否将识别结果保存为图片文件；
      - output_dir (str): 图片的保存路径，当为 None 时，默认设为face_landmark_output。

    - **返回**

      - res (list[dict]): 识别结果的列表，列表元素为 dict, 有以下两个字段：
        - save_path : 可视化图片的保存路径（仅当visualization=True时存在）；
        - data: 图片中每张人脸的关键点坐标

  - ```python
    def set_face_detector_module(face_detector_module):
    ```

    - 设置为人脸关键点检测模型进行人脸检测的底座模型
    - **参数**
      - face_detector_module (class): 人脸检测模型

  - ```python
    def get_face_detector_module():
    ```

    - 获取为人脸关键点检测模型进行人脸检测的底座模型
    - **返回**
      - 当前模型使用的人脸检测模型。

  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=False):
    ```

    - 将模型保存到指定路径，由于人脸关键点检测模型由人脸检测+关键点检测两个模型组成，因此保存后会存在两个子目录，其中`face_landmark`为人脸关键点模型，`detector`为人脸检测模型。
    - **参数**
      - dirname: 存在模型的目录名称
      - model_filename: 模型文件名称，默认为\__model__
      - params_filename: 参数文件名称，默认为\__params__(仅当combined为True时生效)
      - combined: 是否将参数保存到统一的一个文件中

## 四、服务部署

- PaddleHub Serving可以部署一个在线人脸关键点检测服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m face_landmark_localization -p 8866
    ```

  - 这样就完成了一个人脸关键点服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    import cv2
    import base64
    import paddlehub as hub

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')

    # 发送HTTP请求
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/face_landmark_localization"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```

## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

* 1.0.2

* 1.0.3

  移除 fluid api

  * ```shell
    $ hub install face_landmark_localization==1.0.3
    ```
