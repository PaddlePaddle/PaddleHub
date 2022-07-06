# ultra_light_fast_generic_face_detector_1mb_320

|Module Name|ultra_light_fast_generic_face_detector_1mb_320|
| :--- | :---: |
|Category|face detection|
|Network|Ultra-Light-Fast-Generic-Face-Detector-1MB|
|Dataset|WIDER FACEDataset|
|Fine-tuning supported or not|No|
|Module Size|2.6MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/131604811-bce29c3f-66f7-45cb-a388-d739368bfeb9.jpg"   width='50%' hspace='10'/>
    <br />
    </p>

- ### Module Introduction

  - Ultra-Light-Fast-Generic-Face-Detector-1MB is an extreme light-weight model for real-time face detection in low computation power devices. This module is based on Ultra-Light-Fast-Generic-Face-Detector-1MB, trained on WIDER FACEDataset, and can be used for face detection.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install ultra_light_fast_generic_face_detector_1mb_320
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run ultra_light_fast_generic_face_detector_1mb_320 --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
    result = face_detector.face_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = face_detector.face_detection(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def face_detection(images=None,
                       paths=None,
                       batch\_size=1,
                       use_gpu=False,
                       output_dir='face_detector_640_predict_output',
                       visualization=False,
                       confs_threshold=0.5)
    ```

    - Detect all faces in image

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - batch_size (int): the size of batch;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - output_dir (str): save path of images;
      - visualization (bool): Whether to save the results as picture files;
      - confs\_threshold (float): the confidence threshold

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**

      - res (list\[dict\]): results
        - path (str): path for input image
        - data (list): detection results, each element in the list is dict
          - confidence (float): the confidence of the result
          - left (int): the upper left corner x coordinate of the detection box
          - top (int): the upper left corner y coordinate of the detection box
          - right (int): the lower right corner x coordinate of the detection box
          - bottom (int): the lower right corner y coordinate of the detection box
        - save\_path (str): path for saving output image


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

- PaddleHub Serving can deploy an online service of face detection.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m ultra_light_fast_generic_face_detector_1mb_320
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
    url = "http://127.0.0.1:8866/predict/ultra_light_fast_generic_face_detector_1mb_320"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

* 1.1.3

  Remove fluid api

  - ```shell
    $ hub install ultra_light_fast_generic_face_detector_1mb_320==1.1.3
    ```
