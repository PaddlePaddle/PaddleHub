# faster_rcnn_resnet50_fpn_venus

|模型名称|faster_rcnn_resnet50_fpn_venus|
| :--- | :---: | 
|类别|图像 - 目标检测|
|网络|faster_rcnn|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|317MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 

- ### 模型介绍

  - Faster_RCNN是两阶段目标检测器。通过对图像生成候选区域，提取特征，判别特征类别并修正候选框位置。Faster_RCNN整体网络可以分为4个主要内容，一是ResNet-50作为基础卷积层，二是区域生成网络，三是Rol Align，四是检测层。该PaddleHub Module是由800+tag,170w图片，1000w+检测框训练的大规模通用检测模型，在8个数据集上MAP平均提升2.06%，iou=0.5的准确率平均提升1.78%。对比于其他通用检测模型，使用该Module进行finetune，可以更快收敛，达到较优效果。


## 二、安装

- ### 1、环境依赖     

  - paddlepaddle >= 1.6.2    

  - paddlehub >= 1.6.0                            

- ### 2、安装

  - ```shell
    $ hub install faster_rcnn_resnet50_fpn_venus
    ```
  
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run faster_rcnn_resnet50_fpn_venus --input_path "/PATH/TO/IMAGE"
    ```

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    object_detector = hub.Module(name="faster_rcnn_resnet50_fpn_venus")
    result = object_detector.object_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = object_detector.object_detection((paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def context(num_classes=81,
                trainable=True,
                pretrained=True,
                phase='train')
    ```

    - 提取特征，用于迁移学习。

    - **参数**
      - num\_classes (int): 类别数；<br/>
      - trainable (bool): 参数是否可训练；<br/>
      - pretrained (bool): 是否加载预训练模型；<br/>
      - get\_prediction (bool): 可选值为 'train'/'predict'，'train' 用于训练，'predict' 用于预测。
    
    - **返回**
      - inputs (dict): 模型的输入，相应的取值为：
        当phase为'train'时，包含：
          - image (Variable): 图像变量
          - im\_size (Variable): 图像的尺寸
          - im\_info (Variable): 图像缩放信息
          - gt\_class (Variable): 检测框类别
          - gt\_box (Variable): 检测框坐标
          - is\_crowd (Variable): 单个框内是否包含多个物体
        当 phase 为 'predict'时，包含：
          - image (Variable): 图像变量
          - im\_size (Variable): 图像的尺寸
          - im\_info (Variable): 图像缩放信息 
      - outputs (dict): 模型的输出，响应的取值为：
        当 phase 为 'train'时，包含：
          - head_features (Variable): 所提取的特征
          - rpn\_cls\_loss (Variable): 检测框分类损失
          - rpn\_reg\_loss (Variable): 检测框回归损失
          - generate\_proposal\_labels (Variable): 图像信息
        当 phase 为 'predict'时，包含：
          - head_features (Variable): 所提取的特征
          - rois (Variable): 提取的roi
          - bbox\_out (Variable): 预测结果
      - context\_prog (Program): 用于迁移学习的 Program
  
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

- PaddleHub Serving可以部署一个目标检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m faster_rcnn_resnet50_fpn_venus
    ```

  - 

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
    url = "http://127.0.0.1:8866/predict/faster_rcnn_resnet50_fpn_venus"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布
  - ```shell
    $ hub install faster_rcnn_resnet50_fpn_venus==1.0.0
    ```