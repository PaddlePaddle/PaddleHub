# pyramidbox_lite_mobile

|Module Name|pyramidbox_lite_mobile|
| :--- | :---: |
|Category|face detection|
|Network|PyramidBox|
|Dataset|WIDER FACEDataset + 百度自采人脸Dataset|
|Fine-tuning supported or not|No|
|Module Size|7.3MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/131602468-351eb3fb-81e3-4294-ac8e-b49a3a0232cb.jpg"   width='50%' hspace='10'/>
    <br />
    </p>

- ### Module Introduction

  - PyramidBox-Lite是基于2018年百度发表于计算机视觉顶级会议ECCV 2018的论文PyramidBox而研发的轻量级模型，模型基于主干网络FaceBoxes，对于光照、口罩遮挡、表情变化、尺度变化等常见问题具有很强的鲁棒性.该PaddleHub Module是针对于移动端优化过的模型，适合部署于移动端或者边缘检测等算力受限的设备上，并基于WIDER FACE数据集和百度自采人脸数据集进行训练，支持预测，可用于人脸检测.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub]()

- ### 2、Installation

  - ```shell
    $ hub install pyramidbox_lite_mobile
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run pyramidbox_lite_mobile --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    face_detector = hub.Module(name="pyramidbox_lite_mobile")
    result = face_detector.face_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = face_detector.face_detection(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def face_detection(images=None,
                       paths=None,
                       use_gpu=False,
                       output_dir='detection_result',
                       visualization=False,
                       shrink=0.5,
                       confs_threshold=0.6)
    ```

    - 检测输入图片中的所有人脸位置.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - output_dir (str): save path of images;
      - visualization (bool): Whether to save the results as picture files;
      - shrink (float): 用于设置图片的缩放比例，该值越大，则对于输入图片中的小尺寸人脸有更好的检测效果（模型计算成本越高），反之则对于大尺寸人脸有更好的检测效果；<br/>
      - confs\_threshold (float): 置信度的阈值.

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**

      - res (list\[dict\]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability
        - path (str): 原输入图片的路径
        - data (list): 检测结果，list的每一个元素为 dict，各字段为:
          - confidence (float): 识别的置信度
          - left (int): 边界框的左上角x坐标
          - top (int): 边界框的左上角y坐标
          - right (int): 边界框的右下角x坐标
          - bottom (int): 边界框的右下角y坐标


  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - 将模型保存到指定路径.

    - **Parameters**

      - dirname: 存在模型的目录名称； <br/>
      - model\_filename: 模型文件名称，默认为\_\_model\_\_； <br/>
      - params\_filename: Parameters文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)；<br/>
      - combined: 是否将Parameters保存到统一的一个文件中.


## IV.Server Deployment

- PaddleHub Serving can deploy an online service of face detection.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m pyramidbox_lite_mobile
    ```

  - The servitization API is now deployed and the default port number is 8866.

  - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

  - With a configured server, use the following lines of code to send the prediction request and obtain the result

  - ```python
    import requests
    import json
    import cv2
    import base64


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    # Send an HTTP request
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/pyramidbox_lite_mobile"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

* 1.2.0

  - ```shell
    $ hub install pyramidbox_lite_mobile==1.2.0
    ```
