# yolov3_darknet53_vehicles

|Module Name|yolov3_darknet53_vehicles|
| :--- | :---: |
|Category|object detection|
|Network|YOLOv3|
|Dataset|百度自建大规模车辆Dataset|
|Fine-tuning supported or not|No|
|Module Size|238MB|
|Latest update date|2021-03-15|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
     <p align="center">
     <img src="https://user-images.githubusercontent.com/22424850/131529643-70ee93fc-c9f3-40df-a981-901074683beb.jpg"   width='50%' hspace='10'/>
     <br />
     </p>

- ### Module Introduction

  - 车辆检测是城市交通监控中非常重要并且具有挑战性的任务，该任务的难度在于对复杂场景中相对较小的车辆进行精准地定位和分类.该 PaddleHub Module 的网络为 YOLOv3, 其中 backbone 为 DarkNet53，采用百度自建大规模车辆数据集训练得到，支持car (汽车)、truck (卡车)、bus (公交车)、motorbike (摩托车)、tricycle (三轮车)等车型的识别.目前仅支持预测.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub]()

- ### 2、Installation

  - ```shell
    $ hub install yolov3_darknet53_vehicles
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run yolov3_darknet53_vehicles --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)
- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    vehicles_detector = hub.Module(name="yolov3_darknet53_vehicles")
    result = vehicles_detector.object_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = vehicles_detector.object_detection((paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def object_detection(paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='yolov3_vehicles_detect_output',
                         score_thresh=0.2,
                         visualization=True)
    ```

    - 预测API，检测输入图片中的所有车辆的位置.

    - **Parameters**

      - paths (list[str]): image path;
      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - batch_size (int): the size of batch;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - output_dir (str): save path of images;
      - score\_thresh (float): 识别置信度的阈值；<br/>
      - visualization (bool): Whether to save the results as picture files;

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**

      - res (list\[dict\]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability
        - data (list): detection results, each element in the list is dict
          - confidence (float): the confidence of the result
          - label (str): 标签
          - left (int): the upper left corner x coordinate of the detection box
          - top (int): the upper left corner y coordinate of the detection box
          - right (int): the lower right corner x coordinate of the detection box
          - bottom (int): the lower right corner y coordinate of the detection box
        - save\_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)

  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - Save model to specific path

    - **Parameters**

      - dirname: output dir for saving model
      - model\_filename: filename for saving model
      - params\_filename: filename for saving parameters
      - combined: whether save parameters into one file


## IV.Server Deployment

- PaddleHub Serving can deploy an online service of object detection.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m yolov3_darknet53_vehicles
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
    url = "http://127.0.0.1:8866/predict/yolov3_darknet53_vehicles"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

* 1.0.2

  Fix the problem of reading numpy

  - ```shell
    $ hub install yolov3_darknet53_vehicles==1.0.2
    ```
