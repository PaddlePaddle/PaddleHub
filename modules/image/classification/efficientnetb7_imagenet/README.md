# efficientnetb7_imagenet

|模型名称|efficientnetb7_imagenet|
| :--- | :---: |
|类别|图像-图像分类|
|网络|EfficientNet|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|否|
|模型大小|260MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - EfficientNet 是谷歌的开源新模型，是一个轻量级网络，它的主干网络由 MBConv 构成，同时采取了 squeeze-and-excitation 操作对网络结构进行优化。该 PaddleHub Module结构为 EfficientNetB7，基于 ImageNet-2012 数据集训练，接受输入图片大小为 224 x 224 x 3，支持直接通过命令行或者 Python 接口进行预测。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install efficientnetb7_imagenet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run efficientnetb7_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="efficientnetb7_imagenet")
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

      - res (list\[dict\]): 分类结果，列表的每一个元素均为字典，其中 key 为识别的菜品类别，value为置信度。



## 四、服务部署

- PaddleHub Serving可以部署一个图像识别的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m efficientnetb7_imagenet
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
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/efficientnetb7_imagenet"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.1.0

  提升预测性能以及易用性
  - ```shell
    $ hub install efficientnetb7_imagenet==1.1.0
    ```
