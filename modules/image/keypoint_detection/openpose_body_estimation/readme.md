# openpose_body_estimation

| 模型名称            |  openpose_body_estimation  |
| :------------------ | :------------------------: |
| 类别                |      图像-关键点检测       |
| 网络                | two-branch multi-stage CNN |
| 数据集              |      MPII, COCO 2016       |
| 是否支持Fine-tuning |             否             |
| 模型大小            |            185M            |
| 最新更新日期        |         2021-06-28         |
| 数据指标            |             -              |

## 一、模型基本信息

- ### 应用效果展示
  - 人体关键点（左）、模型预测效果（右）
  
    <p align="center">
    <img src="https://user-images.githubusercontent.com/76040149/133232647-011528a1-32f3-416f-a618-17ffbeba6bab.png" height = "300" hspace='10'/> <img src="https://user-images.githubusercontent.com/76040149/133232724-30979d86-8688-483e-abc3-a9159695a56c.png" height = "300" hspace='10'/>
    </p>
    
- ### 模型介绍
  - openpose_body_estimation是基于'Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields'构建的用于肢体关键点检测的模型，该模型可以与openpose_hands_estimation模型联合使用检测肢体和手部关键点。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install openpose_body_estimation
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
   | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run openpose_body_estimation --input_path "/PATH/TO/IMAGE"
    ```
    
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    
    model = hub.Module(name='openpose_body_estimation')
    result = model.predict('/PATH/TO/IMAGE')
    model.save_inference_model('/PATH/TO/SAVE/MODEL')
    
    # PaddleHub示例图片下载方法：
    # wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
    ```
  
- ### 3、API

  - ```python
    def __init__(self, load_checkpoint: str = None):
    ```

    - 模型初始化函数
    - **参数**
      - load_checkpoint(str): 肢体检测模型，用户可以指定自己的模型地址。 默认为None时，会使用PaddleHub提供的默认模型。

  - ```python
    def predict(self,
                img, 
                save_path='openpose_body',  
                visualization=True):
    ```
    
    - 识别输入图片中的所有人肢体关键点。
    - **参数**
      - img (numpy.ndarray|str): 图片数据，使用图片路径或者输入numpy.ndarray，BGR格式；
      - save_path (str): 图片保存路径， 默认为'openpose_body'；
      - visualization (bool): 是否将识别结果保存为图片文件；
    - **返回**
      - res (dict): 识别结果的列表，列表元素为 dict, 有以下两个字段：
        - data : 可视化图片内容(numpy.ndarray，BGR格式);
        - candidate: 图片中所有肢体关键点坐标;
        - subset: 不同的人不同关键点对应的关键点坐标的索引。
    
  - ```python
    def save_inference_model(save_dir):
    ```

    - 将模型保存到指定路径。
    - **参数**
      - save_dir(str): 存在模型的目录名称。


## 四、服务部署

- PaddleHub Serving可以部署一个在线肢体关键点检测服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m openpose_body_estimation -p 8866
    ```

  - 这样就完成了一个肢体关键点服务化API的部署，默认端口号为8866。

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
    org_im = cv2.imread('/PATH/TO/IMAGE')
    data = {'images':[cv2_to_base64(org_im)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/openpose_body_estimation"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    canvas = base64_to_cv2(r.json()["results"]['data'])
    cv2.imwrite('keypoint_body.png', canvas)
    ```

## 五、更新历史

* 1.0.0

  初始发布
  
  * ```shell
    $ hub install openpose_body_estimation==1.0.0
    ```

