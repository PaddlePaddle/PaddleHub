# german_ocr_db_crnn_mobile

|模型名称|german_ocr_db_crnn_mobile|
| :--- | :---: |
|类别|图像-文字识别|
|网络|Differentiable Binarization+CRNN|
|数据集|icdar2015数据集|
|是否支持Fine-tuning|否|
|模型大小|3.8MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/133761772-8c47f25f-0d95-45b4-8075-867dbbd14c86.jpg"  width="80%" hspace='10'/> <br />
    </p>

- ### 模型介绍

  - german_ocr_db_crnn_mobile Module用于识别图片当中的德文。其基于chinese_text_detection_db_mobile检测得到的文本框，继续识别文本框中的德文文字。最终识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。其是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。该Module是一个识别德文的轻量级OCR模型，支持直接预测。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.2  

  - paddlehub >= 2.0.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install german_ocr_db_crnn_mobile
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run german_ocr_db_crnn_mobile --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)


- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    ocr = hub.Module(name="german_ocr_db_crnn_mobile", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def __init__(text_detector_module=None, enable_mkldnn=False)
    ```

    - 构造GenmanOCRDBCRNNMobile对象

    - **参数**

      - text_detector_module(str): 文字检测PaddleHub Module名字，如设置为None，则默认使用[chinese_text_detection_db_mobile Module](../chinese_text_detection_db_mobile/)。其作用为检测图片当中的文本。<br/>
      - enable_mkldnn(bool): 是否开启mkldnn加速CPU计算。该参数仅在CPU运行下设置有效。默认为False。

  - ```python
    def recognize_text(images=[],
                       paths=[],
                       use_gpu=False,
                       output_dir='ocr_result',
                       visualization=False,
                       box_thresh=0.5,
                       text_thresh=0.5,
                       angle_classification_thresh=0.9)
    ```

    - 预测API，检测输入图片中的所有德文文本的位置。

    - **参数**

      - paths (list\[str\]): 图片的路径； <br/>
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式； <br/>
      - use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量** <br/>
      - box\_thresh (float): 检测文本框置信度的阈值； <br/>
      - text\_thresh (float): 识别德文文本置信度的阈值； <br/>
      - angle_classification_thresh(float): 文本角度分类置信度的阈值 <br/>
      - visualization (bool): 是否将识别结果保存为图片文件； <br/>
      - output\_dir (str): 图片的保存路径，默认设为 ocr\_result；

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
        - data (list\[dict\]): 识别文本结果，列表中每一个元素为 dict，各字段为：
          - text(str): 识别得到的文本
          - confidence(float): 识别文本结果置信度
          - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
      如果无识别结果则data为\[\]
        - save_path (str, optional): 识别结果的保存路径，如不保存图片则save_path为''



## 四、服务部署

- PaddleHub Serving 可以部署一个目标检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m german_ocr_db_crnn_mobile
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
    url = "http://127.0.0.1:8866/predict/german_ocr_db_crnn_mobile"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```

## 五、更新历史

* 1.0.0

  初始发布

* 1.1.0

  优化模型
  - ```shell
    $ hub install german_ocr_db_crnn_mobile==1.1.0
    ```
