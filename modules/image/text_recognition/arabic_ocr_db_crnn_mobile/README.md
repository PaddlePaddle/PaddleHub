# arabic_ocr_db_crnn_mobile

|模型名称|arabic_ocr_db_crnn_mobile|
| :--- | :---: |
|类别|图像-文字识别|
|网络|Differentiable Binarization+CRNN|
|数据集|icdar2015数据集|
|是否支持Fine-tuning|否|
|最新更新日期|2021-12-2|
|数据指标|-|


## 一、模型基本信息

- ### 模型介绍

  - arabic_ocr_db_crnn_mobile Module用于识别图片当中的阿拉伯文字，包括阿拉伯文、波斯文、维吾尔文。其基于multi_languages_ocr_db_crnn检测得到的文本框，继续识别文本框中的阿拉伯文文字。最终识别文字算法采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。其是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。该Module是一个识别阿拉伯文的轻量级OCR模型，支持直接预测。

  - 更多详情参考：
    - [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
    - [An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.2  

  - paddlehub >= 2.0.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install arabic_ocr_db_crnn_mobile
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run arabic_ocr_db_crnn_mobile --input_path "/PATH/TO/IMAGE"
    $ hub run arabic_ocr_db_crnn_mobile --input_path "/PATH/TO/IMAGE" --det True --rec True --use_angle_cls True  --box_thresh 0.7 --angle_classification_thresh 0.8 --visualization True
    ```
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    ocr = hub.Module(name="arabic_ocr_db_crnn_mobile", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def __init__(self,
                 det=True,
                 rec=True,
                 use_angle_cls=False,
                 enable_mkldnn=False,  
                 use_gpu=False,
                 box_thresh=0.6,
                 angle_classification_thresh=0.9)
    ```

    - 构造ArabicOCRDBCRNNMobile对象

    - **参数**
      - det(bool): 是否开启文字检测。默认为True。
      - rec(bool): 是否开启文字识别。默认为True。
      - use_angle_cls(bool): 是否开启方向分类, 用于设置使用方向分类器识别180度旋转文字。默认为False。
      - enable_mkldnn(bool): 是否开启mkldnn加速CPU计算。该参数仅在CPU运行下设置有效。默认为False。
      - use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**
      - box\_thresh (float): 检测文本框置信度的阈值；
      - angle_classification_thresh(float): 文本方向分类置信度的阈值


  - ```python
    def recognize_text(images=[],
                       paths=[],
                       output_dir='ocr_result',
                       visualization=False)
    ```

    - 预测API，检测输入图片中的所有文本的位置和识别文本结果。

    - **参数**

      - paths (list\[str\]): 图片的路径；
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      - output\_dir (str): 图片的保存路径，默认设为 ocr\_result；
      - visualization (bool): 是否将识别结果保存为图片文件, 仅有检测开启时有效, 默认为False；

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
        - data (list\[dict\]): 识别文本结果，列表中每一个元素为 dict，各字段为：
          - text(str): 识别得到的文本
          - confidence(float): 识别文本结果置信度
          - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标，如果无识别结果则data为\[\]
          - orientation(str): 分类的方向，仅在只有方向分类开启时输出
          - score(float): 分类的得分，仅在只有方向分类开启时输出
        - save_path (str, optional): 识别结果的保存路径，如不保存图片则save_path为''


## 四、服务部署

- PaddleHub Serving 可以部署一个目标检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m arabic_ocr_db_crnn_mobile
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
    url = "http://127.0.0.1:8866/predict/arabic_ocr_db_crnn_mobile"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```

## 五、更新历史

* 1.0.0

  初始发布
  - ```shell
    $ hub install arabic_ocr_db_crnn_mobile==1.0.0
    ```
