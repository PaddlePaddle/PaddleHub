# multi_languages_ocr

|模型名称|multi_languages_ocr|
| :--- | :---: |
|类别|图像-文字识别|
|网络|Differentiable Binarization+RCNN|
|数据集|icdar2015数据集|
|是否支持Fine-tuning|否|
|模型大小|xxxM|
|最新更新日期|2021-11-24|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - [OCR文字识别场景在线体验](https://www.paddlepaddle.org.cn/hub/scene/ocr)
  - 样例结果示例：
<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133097562-d8c9abd1-6c70-4d93-809f-fa4735764836.png"  width = "600" hspace='10'/> <br />
</p>

- ### 模型介绍

  - multi_languages_ocr Module用于识别图片当中的汉字。其基于PaddleOCR模块，检测得到的文本框，识别文本框中的文字,再对检测文本框进行角度分类。最终识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。该Module是一个支持80种语言的检测和识别，支持轻量高精度英文模型检测识别，支持直接预测。


<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133098254-7c642826-d6d7-4dd0-986e-371622337867.png" width = "300" height = "450"  hspace='10'/> <br />
</p>

  - 更多详情参考：[An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)



## 二、安装

- ### 1、环境依赖  

  - PaddlePaddle >= 1.8.0  

  - PaddleOCR >= 2.0.1   | [如何安装PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/quickstart.md#1)

  - PaddleHub >= 1.8.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

  - shapely

  - pyclipper

  - ```shell
    $ pip install shapely pyclipper
    ```
  - **该Module依赖于第三方库shapely和pyclipper，使用该Module之前，请先安装shapely和pyclipper。**  

- ### 2、安装

  - ```shell
    $ hub install multi_languages_ocr
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run multi_languages_ocr --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    ocr = hub.Module(name="multi_languages_ocr", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效
    result = ocr.predict(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result = ocr.predict(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def _initialize(self,
                lang="ch",
                det=True,
                rec=True,
                use_angle_cls=False,
                use_gpu=False,
                enable_mkldnn=False)
    ```

    - 构造MultiLangOCR对象

    - **参数**
      - lang(bool): 多语言选择。默认为ch。
      - det(bool): 是否开启检测。默认为True。
      - rec(bool): 是否开启识别。默认为True。
      - use_angle_cls(bool): 是否开启方向判断。默认为False。
      - use_gpu(bool): 是否开启gpu。默认为False。
      - enable_mkldnn(bool): 是否开启mkldnn加速CPU计算。该参数仅在CPU运行下设置有效。默认为False。


  - ```python
    def predict(images=[],
                paths=[],
                output_dir='ocr_result',
                visualization=False)
    ```

    - 预测API，检测输入图片中的所有中文文本的位置。

    - **参数**

      - paths (list\[str\]): 图片的路径；
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      - visualization (bool): 是否将识别结果保存为图片文件；
      - output\_dir (str): 图片的保存路径，默认设为 ocr\_result；

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
        - data (list\[dict\]): 识别文本结果，列表中每一个元素为 dict，各字段为：
        - save_path (str, optional): 识别结果的保存路径，如不保存图片则save_path为''


## 四、服务部署

- PaddleHub Serving 可以部署一个目标检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m multi_languages_ocr
    ```

  - 这样就完成了一个目标检测的服务化API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/chinese_ocr_db_crnn_mobile"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```
