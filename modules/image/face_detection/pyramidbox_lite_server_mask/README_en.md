# pyramidbox_lite_server_mask

|Module Name|pyramidbox_lite_server_mask|
| :--- | :---: |
|Category|face detection|
|Network|PyramidBox|
|Dataset|WIDER FACEDataset + Baidu Face Dataset|
|Fine-tuning supported or not|No|
|Module Size|1.2MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/131603304-690a2e3b-9f24-42f6-9297-a12ada884191.jpg"   width='50%' hspace='10'/>
    <br />
    </p>

- ### Module Introduction

  - PyramidBox-Lite is a light-weight model based  on PyramidBox proposed by Baidu in ECCV 2018. This model has solid robustness against interferences such as light and scale variation. This module is based on PyramidBox, trained on WIDER FACE Dataset and Baidu Face Dataset, and can be used for mask detection.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install pyramidbox_lite_server_mask
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run pyramidbox_lite_server_mask --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    mask_detector = hub.Module(name="pyramidbox_lite_server_mask")
    result = mask_detector.face_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = mask_detector.face_detection(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API


  - ```python
    def face_detection(images=None,
                       paths=None,
                       batch_size=1,
                       use_gpu=False,
                       visualization=False,
                       output_dir='detection_result',
                       use_multi_scale=False,
                       shrink=0.5,
                       confs_threshold=0.6)
    ```

    - Detect all faces in image, and judge the existence of mask.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - batch_size (int): the size of batch;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - visualization (bool): Whether to save the results as picture files;
      - output_dir (str): save path of images;
      - use\_multi\_scale (bool) : whether to detect across multiple scales;
      - shrink (float): the scale to resize image
      - confs\_threshold (float): the confidence threshold

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**

      - res (list\[dict\]): results
        - path (str): path for input image
        - data (list): detection results, each element in the list is dict
          - label (str): 'NO MASK' or 'MASK'；
          - confidence (float): the confidence of the result
          - left (int): the upper left corner x coordinate of the detection box
          - top (int): the upper left corner y coordinate of the detection box
          - right (int): the lower right corner x coordinate of the detection box
          - bottom (int): the lower right corner y coordinate of the detection box



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
    $ hub serving start -m pyramidbox_lite_server_mask
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
    url = "http://127.0.0.1:8866/predict/pyramidbox_lite_server_mask"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```
## V.Paddle Lite Deployment
- ### Save model demo
  - ```python
    import paddlehub as hub
    pyramidbox_lite_server_mask = hub.Module(name="pyramidbox_lite_server_mask")

    # save model in directory named test_program
    pyramidbox_lite_server_mask.save_inference_model(dirname="test_program")
    ```


- ### transform model

  - The model downloaded from paddlehub is a prediction model. If we want to deploy it in mobile device, we can use OPT tool provided by PaddleLite to transform the model. For more information, please refer to [OPT tool](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html))

- ### Deploy the model with Paddle Lite
  - Please refer to[Paddle-Lite mask detection model deployment demo](https://github.com/PaddlePaddle/Paddle-Lite/tree/develop/lite/demo/cxx)

## V.Release Note

* 1.0.0

  First release

* 1.3.2

  Remove fluid api

  - ```shell
    $ hub install pyramidbox_lite_server_mask==1.3.2
    ```
