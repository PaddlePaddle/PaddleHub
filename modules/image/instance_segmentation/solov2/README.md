# solov2

| 模型名称            |    solov2     |
| :------------------ | :-----------: |
| 类别                | 图像-实例分割 |
| 网络                |       -       |
| 数据集              |   COCO2014    |
| 是否支持Fine-tuning |      否       |
| 模型大小            |     165M      |
| 最新更新日期        |  2021-02-26   |
| 数据指标            |       -       |

## 一、模型基本信息

- ### 应用效果展示

<div align="center">
<img src="https://user-images.githubusercontent.com/76040149/133250741-83040204-acfc-4348-af90-acac74f40cd8.png"   height = "300" />
</div>

- ### 模型介绍
  - solov2是基于"SOLOv2: Dynamic, Faster and Stronger"实现的快速实例分割的模型。该模型基于SOLOV1, 并且针对mask的检测效果和运行效率进行改进，在实例分割任务中表现优秀。相对语义分割，实例分割需要标注出图上同一物体的不同个体。
  

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0
  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)
  
- ### 2、安装

  - ```shell
    $ hub install solov2
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
   | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run openpose_hands_estimation --input_path "/PATH/TO/IMAGE"
    ```
    
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import cv2
    import paddlehub as hub
    
    img = cv2.imread('/PATH/TO/IMAGE')
    model = hub.Module(name='solov2', use_gpu=False)
    output = model.predict(image=img, visualization=False)
    ```
  
- ### 3、API

  - ```python
    def predict(image: Union[str, np.ndarray],
                threshold: float = 0.5,
                visualization: bool = False,
                save_dir: str = 'solov2_result'):
    ```
    
    - 预测API，实例分割。
    - **参数**
      - image (Union[str, np.ndarray]): 图片路径或者图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
      - threshold (float): 检测模型输出结果中，预测得分低于该阈值的框将被滤除，默认值为0.5；
      - visualization (bool): 是否将可视化图片保存；
      - save_dir (str): 保存图片到路径， 默认为"solov2_result"。
    - **返回**
      - res (dict): 识别结果，关键字有 'segm', 'label', 'score'对应的取值为：
        - segm (np.ndarray): 实例分割结果,取值为0或1。0表示背景，1为实例；
        - label (list): 实例分割结果类别id；
        - score (list):实例分割结果类别得分；s


## 四、服务部署

- PaddleHub Serving可以部署一个实例分割的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m solov2 -p 8866
    ```

  - 这样就完成了一个实例分割的在线服务API的部署，默认端口号为8866。

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
    h, w, c = org_im.shape
    data = {'images':[cv2_to_base64(org_im)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/solov2"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    seg = base64.b64decode(r.json()["results"]['segm'].encode('utf8'))
    seg = np.fromstring(seg, dtype=np.int32).reshape((-1, h, w))
    
    label = base64.b64decode(r.json()["results"]['label'].encode('utf8'))
    label = np.fromstring(label, dtype=np.int64)
    
    score = base64.b64decode(r.json()["results"]['score'].encode('utf8'))
    score = np.fromstring(score, dtype=np.float32)
    
    print('seg', seg)
    print('label', label)
    print('score', score)
    ```

## 五、更新历史

* 1.0.0

  初始发布

  * ```shell
    $ hub install hand_pose_localization==1.0.0
    ```

    
