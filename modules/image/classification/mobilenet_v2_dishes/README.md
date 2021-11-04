# mobilenet_v2_dishes

|模型名称|mobilenet_v2_dishes|
| :--- | :---: |
|类别|图像-图像分类|
|网络|MobileNet_v2|
|数据集|百度自建菜品数据集|
|是否支持Fine-tuning|否|
|模型大小|52MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - MobileNet V2 是一个轻量化的卷积神经网络，它在 MobileNet 的基础上，做了 Inverted Residuals 和 Linear bottlenecks 这两大改进。该 PaddleHub Module 是在百度自建菜品数据集上训练得到的，可用于图像分类和特征提取，当前已支持8416种菜品的分类识别。

<p align="center">
<img src="http://bj.bcebos.com/ibox-thumbnail98/e7b22762cf42ab0e1e1fab6b8720938b?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2020-04-08T11%3A49%3A16Z%2F1800%2F%2Faf385f56da3c8ee1298588939d93533a72203c079ae1187affa2da555b9898ea" width = "800"  hspace='10'/> <br />
</p>

  - 更多详情参考：[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install mobilenet_v2_dishes
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run mobilenet_v2_dishes --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现菜品分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="mobilenet_v2_dishes")
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

- PaddleHub Serving可以部署一个菜品分类的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m mobilenet_v2_dishes
    ```

  - 这样就完成了一个菜品分类的在线服务的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/mobilenet_v2_dishes"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install mobilenet_v2_dishes==1.0.0
    ```
