# resnet50_vd_animals

|模型名称|resnet50_vd_animals|
| :--- | :---: |
|类别|图像-图像分类|
|网络|ResNet50_vd|
|数据集|百度自建动物数据集|
|是否支持Fine-tuning|否|
|模型大小|154MB|
|指标|-|
|最新更新日期|2021-02-26|


## 一、模型基本信息


- ### 模型介绍

    - ResNet-vd 其实就是 ResNet-D，是ResNet 原始结构的变种，可用于图像分类和特征提取。该 PaddleHub Module 采用百度自建动物数据集训练得到，支持7978种动物的分类识别。

    - 模型的详情可参考[论文](https://arxiv.org/pdf/1812.01187.pdf)

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、安装

    - ```shell
      $ hub install resnet50_vd_animals
      ```
    - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

    - ```
      hub run resnet50_vd_animals --input_path "/PATH/TO/IMAGE"
      ```

- ### 2、预测代码示例

    - ```python
      import paddlehub as hub
      import cv2

      classifier = hub.Module(name="resnet50_vd_animals")

      result = classifier.classification(images=[cv2.imread('/PATH/TO/IMAGE')])
      # or
      # result = classifier.classification(paths=['/PATH/TO/IMAGE'])
      ```
- ### 3、API

    - ```python
      def get_expected_image_width()
      ```

        - 返回预处理的图片宽度，也就是224。

    - ```python
      def get_expected_image_height()
      ```

        - 返回预处理的图片高度，也就是224。

    - ```python
      def get_pretrained_images_mean()
      ```

        - 返回预处理的图片均值，也就是 \[0.485, 0.456, 0.406\]。

    - ```python
      def get_pretrained_images_std()
      ```

        - 返回预处理的图片标准差，也就是 \[0.229, 0.224, 0.225\]。


    - ```python
      def classification(images=None,
                         paths=None,
                         batch_size=1,
                         use_gpu=False,
                         top_k=1):
      ```

        - **参数**

            * images (list\[numpy.ndarray\]): 图片数据，每一个图片数据的shape 均为 \[H, W, C\]，颜色空间为 BGR；
            * paths (list\[str\]): 图片的路径；
            * batch\_size (int): batch 的大小；
            * use\_gpu (bool): 是否使用 GPU 来预测；
            * top\_k (int): 返回预测结果的前 k 个。

        - **返回**

            -   res (list\[dict\]): 分类结果，列表的每一个元素均为字典，其中 key 为识别动物的类别，value为置信度。

    - ```python
      def save_inference_model(dirname,
                               model_filename=None,
                               params_filename=None,
                               combined=True)
      ```

        - 将模型保存到指定路径。

        - **参数**

            * dirname: 存在模型的目录名称
            * model_filename: 模型文件名称，默认为\_\_model\_\_
            * params_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)
            * combined: 是否将参数保存到统一的一个文件中


## 四、服务部署

- PaddleHub Serving可以部署一个在线动物识别服务。

- ### 第一步：启动PaddleHub Serving

    - 运行启动命令：

        - ```shell
          $ hub serving start -m resnet50_vd_animals
          ```

        - 这样就完成了一个在线动物识别服务化API的部署，默认端口号为8866。

        - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

- 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
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
    url = "http://127.0.0.1:8866/predict/resnet50_vd_animals"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```

## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  移除 fluid api

  - ```shell
    $ hub install resnet50_vd_animals==1.0.1
    ```
