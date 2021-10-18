# human_pose_estimation_resnet50_mpii

| 模型名称            | human_pose_estimation_resnet50_mpii |
| :------------------ | :---------------------------------: |
| 类别                |           图像-关键点检测           |
| 网络                |            Pose_Resnet50            |
| 数据集              |                MPII                 |
| 是否支持Fine-tuning |                 否                  |
| 模型大小            |                121M                 |
| 最新更新日期        |             2021-02-26              |
| 数据指标            |                                     |

## 一、模型基本信息

- ### 应用效果展示

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133231581-1dffc391-652d-417f-b8ad-a3c22b8092e8.jpg" width="300">
</p>
  
- ### 模型介绍
  - 人体骨骼关键点检测(Pose Estimation) 是计算机视觉的基础算法之一，在很多cv任务中起到了基础性的作用，如行为识别、人物跟踪、步态识别等领域。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 1.6.2

  - paddlehub >= 1.6.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install human_pose_estimation_resnet50_mpii
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
   | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run human_pose_estimation_resnet50_mpii --input_path "/PATH/TO/IMAGE"
    ```
    
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import cv2
    import paddlehub as hub
    
    pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")
    
    result = pose_estimation.keypoint_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = pose_estimation.keypoint_detection(paths=['/PATH/TO/IMAGE'])
    
    # PaddleHub示例图片下载方法：
    # wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
    ```
  
- ### 3、API

  - ```python
    def keypoint_detection(images=None,
                           paths=None,
                           batch_size=1,
                           use_gpu=False,
                           output_dir='output_pose',
                           visualization=False):
    ```
    
    - 预测API，识别出人体骨骼关键点。
    - **参数**
      - images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]；
      - paths (list[str]): 图片的路径；
      - batch_size (int): batch 的大小；
      - use_gpu (bool): 是否使用 GPU；
      - visualization (bool): 是否将识别结果保存为图片文件；
      - output_dir (str): 图片的保存路径，默认设为 output_pose。
    - **返回**
      - res (list): 识别元素的列表，列表元素为 dict，关键字为 'path', 'data'，相应的取值为：
        - path (str): 原图的路径；
        - data (OrderedDict): 人体骨骼关键点的坐标。
    
  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True):
    ```
  
    - 将模型保存到指定路径。
  
    - **参数**
  
      - dirname: 存在模型的目录名称
      - model_filename: 模型文件名称，默认为__model__
      - params_filename: 参数文件名称，默认为__params__(仅当combined为True时生效)
      - combined: 是否将参数保存到统一的一个文件中


## 四、服务部署

- PaddleHub Serving可以部署一个在线人体骨骼关键点识别服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m human_pose_estimation_resnet50_mpii -p 8866
    ```

  - 这样就完成了一个人体骨骼关键点识别的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/human_pose_estimation_resnet50_mpii"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    # 打印预测结果
    print(r.json()["results"])
    
    # r.json()['results']即为keypoint_detection函数返回的结果
    ```

## 五、更新历史

* 1.0.0

* 1.1.0

* 1.1.1

  * ```shell
    $ hub install human_pose_estimation_resnet50_mpii==1.1.1
    ```
