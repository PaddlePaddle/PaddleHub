# openpose_hands_estimation

| 模型名称            | hand_pose_localization |
| :------------------ | :--------------------: |
| 类别                |    图像-关键点检测     |
| 网络                |           -            |
| 数据集              |       MPII, NZSL       |
| 是否支持Fine-tuning |           否           |
| 模型大小            |          130M          |
| 最新更新日期        |       2021-06-02       |
| 数据指标            |           -            |

## 一、模型基本信息

- ### 应用效果展示
  - 手部关键点展示（左）、预测效果（右）
<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133233743-dc6e3aaf-27fd-4f7d-95be-21c9383a2ea1.png" height="300"><img src="https://user-images.githubusercontent.com/76040149/133234189-f7a47940-2be2-445c-8043-b490b5402e15.png" height="300">
</p>

- ### 模型介绍
  - penpose_hands_estimation是基于 'Hand Keypoint Detection in Single Images using Multiview Bootstrapping' 构建的用于手部关键点检测的模型。
  

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0
  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)
  - scikit-image
  - scipy
  - ```shell
    $ pip install scikit-image scipy
    ```

- ### 2、安装

  - ```shell
    $ hub install openpose_hands_estimation
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
   | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run openpose_hands_estimation --input_path "/PATH/TO/IMAGE"
    ```
    
  - Note：本模型先识别人体关键点以确定2个手的位置，再识别手部关键点；输入图片建议为半身照或全身照，手部没有遮挡；本模型需要用到openpose_body_estimation，若未安装则推理前会自动安装
    
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    
    model = hub.Module(name='openpose_hands_estimation')
    result = model.predict('/PATH/TO/IMAGE')
    model.save_inference_model('/PATH/TO/SAVE/MODEL')
    ```
  
- ### 3、API

  - ```python
    def __init__(load_checkpoint: str = None):
    ```

    - **参数**
      - load_checkpoint(str): 手部检测模型，用户可以指定自己的模型地址。 默认为None时，会使用PaddleHub提供的默认模型。

  - ```python
    def predict(img, 
                save_path='openpose_hand', 
                scale=[0.5, 1.0, 1.5, 2.0], 
                visualization=True):
    ```

    - 识别输入图片中的所有人手部关键点。
    - **参数**
      - img (numpy.ndarray|str): 图片数据，使用图片路径或者输入numpy.ndarray，BGR格式；
      - save_path (str): 图片保存路径， 默认为openpose_hand；
      - scale (list): 搜索关键点时使用图片的不同尺度；
      - visualization (bool): 是否将识别结果保存为图片文件；
    - **返回**
      - res (dict): 识别结果的列表，列表元素为 dict, 有以下两个字段：
        - data : 可视化图片内容（numpy.ndarray，BGR格式）；
        - all_hand_peaks: 图片中手部关键点坐标

  - ```python
    def save_inference_model(save_dir):
    ```

    - 将模型保存到指定路径。
    - **参数**
      - save_dir(str): 存放模型的目录名称

## 四、服务部署

- PaddleHub Serving可以部署一个在线手部关键点检测服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m openpose_hands_estimation -p 8866
    ```

  - 这样就完成了一个人体手部关键点检测的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/openpose_hands_estimation"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    canvas = base64_to_cv2(r.json()["results"]["data"])
    cv2.imwrite('keypoint.png', canvas)
    ```

## 五、更新历史

* 1.0.0

  初始发布

  * ```shell
    $ hub install hand_pose_localization==1.0.0
    ```

    
