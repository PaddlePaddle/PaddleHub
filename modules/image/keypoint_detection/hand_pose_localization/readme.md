# hand_pose_localization

| 模型名称            | hand_pose_localization |
| :------------------ | :--------------------: |
| 类别                |    图像-关键点检测     |
| 网络                |                        |
| 数据集              |       MPII, NZSL       |
| 是否支持Fine-tuning |           否           |
| 模型大小            |          130M          |
| 最新更新日期        |       2021-06-02       |
| 数据指标            |                        |

## 一、模型基本信息

- ### 应用效果展示

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133246893-f47cfdce-b9c1-490b-b1de-f837b61caf18.png" align="center" width="500">
</p>
  
- ### 模型介绍
  - openpose 手部关键点检测模型。更多详情请参考：[openpose开源项目](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install hand_pose_localization
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
   | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- 本模型不支持命令行预测

- ### 1、预测代码示例

  - ```python
    import cv2
    import paddlehub as hub
    
    # use_gpu：是否使用GPU进行预测
    model = hub.Module(name='hand_pose_localization', use_gpu=False)
    
    # 调用关键点检测API
    result = model.keypoint_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.keypoint_detection(paths=['/PATH/TO/IMAGE'])
    
    # 打印预测结果
    print(result)
    ```
  
- ### 2、API

  - ```python
    def keypoint_detection(images=None,
                           paths=None,
                           batch_size=1,
                           output_dir='output',
                           visualization=False)：
    ```
    
    - 预测API，识别出人体手部关键点。
    - **参数**
      - images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C], 默认设为 None；
      - paths (list[str]): 图片的路径, 默认设为 None；
      - batch_size (int): batch 的大小，默认设为 1；
      - visualization (bool): 是否将识别结果保存为图片文件，默认设为 False；
      - output_dir (str): 图片的保存路径，默认设为 output。
    - **返回**
      - res (list[list[list[int](https://www.paddlepaddle.org.cn/hubdetail?name=hand_pose_localization&en_category=KeyPointDetection)]]): 每张图片识别到的21个手部关键点组成的列表，每个关键点的格式为[x, y](https://www.paddlepaddle.org.cn/hubdetail?name=hand_pose_localization&en_category=KeyPointDetection)，若有关键点未识别到则为None

## 四、服务部署

- PaddleHub Serving可以部署一个在线人体手部关键点检测服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m hand_pose_localization -p 8866
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
    
    # 图片Base64编码函数
    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')
    
    # 发送HTTP请求
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/hand_pose_localization"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    # 打印预测结果
    print(r.json()["results"])
    ```

## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  适配paddlehub 2.0

  * ```shell
    $ hub install hand_pose_localization==1.0.1
    ```

    
