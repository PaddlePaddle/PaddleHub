# esnet_x0_5_imagenet

|模型名称|esnet_x0_5_imagenet|
| :--- | :---: |
|类别|图像-图像分类|
|网络|ESNet|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|否|
|模型大小|12 MB|
|最新更新日期|2022-04-02|
|数据指标|Acc|


## 一、模型基本信息



- ### 模型介绍

  - ESNet(Enhanced ShuffleNet)是百度自研的一个轻量级网络，该网络在 ShuffleNetV2 的基础上融合了 MobileNetV3、GhostNet、PPLCNet 的优点，组合成了一个在 ARM 设备上速度更快、精度更高的网络，由于其出色的表现，所以在 PaddleDetection 推出的 [PP-PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet) 使用了该模型做 backbone，配合更强的目标检测算法，最终的指标一举刷新了目标检测模型在 ARM 设备上的 SOTA 指标。该模型为模型规模参数scale为x0.5下的ESNet模型。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install esnet_x0_5_imagenet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run esnet_x0_5_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="esnet_x0_5_imagenet")
    result = classifier.classification(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = classifier.classification(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API


  - ```python
    def classification(images=None,
                       paths=None,
                       batch_size=1,
                       use_gpu=False,
                       top_k=1):
    ```
    - 分类接口API。
    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，每一个图片数据的shape 均为 \[H, W, C\]，颜色空间为 BGR； <br/>
      - paths (list\[str\]): 图片的路径； <br/>
      - batch\_size (int): batch 的大小；<br/>
      - use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量** <br/>
      - top\_k (int): 返回预测结果的前 k 个。

    - **返回**

      - res (list\[dict\]): 分类结果，列表的每一个元素均为字典，其中 key 包括'class_ids'（种类索引）, 'scores'（置信度） 和 'label_names'（种类名称）


## 四、服务部署

- PaddleHub Serving可以部署一个图像识别的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m esnet_x0_5_imagenet
    ```

  - 这样就完成了一个图像识别的在线服务的部署，默认端口号为8866。

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
    headers = {"Content-type": "application/json"\}
    url = "http://127.0.0.1:8866/predict/esnet_x0_5_imagenet"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install esnet_x0_5_imagenet==1.0.0
    ```
