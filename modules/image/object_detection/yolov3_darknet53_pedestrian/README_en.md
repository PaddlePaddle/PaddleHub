# yolov3_darknet53_pedestrian

|Module Name|yolov3_darknet53_pedestrian|
| :--- | :---: |
|Category|object detection|
|Network|YOLOv3|
|Dataset|Baidu Pedestrian Dataset|
|Fine-tuning supported or not|No|
|Module Size|238MB|
|Latest update date|2021-03-15|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
     <p align="center">
     <img src="https://user-images.githubusercontent.com/22424850/131492636-714c697c-3275-4c8c-a83a-cf971a91ba98.jpg"   width='50%' hspace='10'/>
     <br />
     </p>

- ### Module Introduction

  - YOLOv3 is a one-stage detector proposed by Joseph Redmon and Ali Farhadi, which can reach comparable accuracy but twice as fast as traditional methods. This module is based on YOLOv3, trained on Baidu Pedestrian Dataset, and can be used for pedestrian detection.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install yolov3_darknet53_pedestrian
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run yolov3_darknet53_pedestrian --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)
- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    pedestrian_detector = hub.Module(name="yolov3_darknet53_pedestrian")
    result = pedestrian_detector.object_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = pedestrian_detector.object_detection(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def object_detection(paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='yolov3_pedestrian_detect_output',
                         score_thresh=0.2,
                         visualization=True)  
    ```

    - Detection API, detect positions of all pedestrian in image

    - **Parameters**

      - paths (list[str]): image path;
      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - batch_size (int): the size of batch;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - output_dir (str): save path of images;
      - score\_thresh (float): confidence threshold；<br/>
      - visualization (bool): Whether to save the results as picture files;

      **NOTE:** choose one parameter to provide data from paths and images


    - **Return**

      - res (list\[dict\]): results
        - data (list): detection results, each element in the list is dict
          - confidence (float): the confidence of the result
          - label (str): label
          - left (int): the upper left corner x coordinate of the detection box
          - top (int): the upper left corner y coordinate of the detection box
          - right (int): the lower right corner x coordinate of the detection box
          - bottom (int): the lower right corner y coordinate of the detection box
        - save\_path (str, optional): output path for saving results

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
    $ hub serving start -m yolov3_darknet53_pedestrian
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
    url = "http://127.0.0.1:8866/predict/yolov3_darknet53_pedestrian"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release  

* 1.0.2

   Fix the problem of reading numpy

* 1.0.3

   Remove fluid api

  - ```shell
    $ hub install yolov3_darknet53_pedestrian==1.0.3
    ```
