# chinese_ocr_db_crnn_mobile

|     Module Name      |  chinese_ocr_db_crnn_mobile  |
|  :------------------ | :------------: |
|       Category       | image-text_recognition |
|         Network      |     Differentiable Binarization+RCNN     |
|         Dataset      | icdar2015 |
| Fine-tuning supported or not |      No       |
|     Module Size      |       16M       |
| Latest update date   |   2021-02-26   |
|   Data indicators    |       -        |


## I. Basic Information of Module

- ### Application Effect Display
  - [Online experience in OCR text recognition scenarios](https://www.paddlepaddle.org.cn/hub/scene/ocr)
  - Example result:
<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133097562-d8c9abd1-6c70-4d93-809f-fa4735764836.png"  width = "600" hspace='10'/> <br />
</p>

- ### Module Introduction

  - chinese_ocr_db_crnn_mobile Module is used to identify Chinese characters in pictures. Get the text box after using [chinese_text_detection_db_mobile Module](../chinese_text_detection_db_mobile/), identify the Chinese characters in the text box, and then do angle classification to the detection text box. CRNN(Convolutional Recurrent Neural Network) is adopted as the final recognition algorithm. This Module is an ultra-lightweight Chinese OCR model that supports direct prediction.


<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133098254-7c642826-d6d7-4dd0-986e-371622337867.png" width = "300" height = "450"  hspace='10'/> <br />
</p>

  - For more information, please refer to:[An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)



## II. Installation

- ### 1、Environmental dependence  

  - paddlepaddle >= 1.7.2  

  - paddlehub >= 1.6.0   | [How to install PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

  - shapely

  - pyclipper

  - ```shell
    $ pip install shapely pyclipper
    ```
  - **This Module relies on the third-party libraries shapely and pyclipper. Please install shapely and pyclipper before using this Module.**  

- ### 2、Installation

  - ```shell
    $ hub install chinese_ocr_db_crnn_mobile
    ```
  - If you have problems during installation, please refer to:[windows_quickstart](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [linux_quickstart](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [mac_quickstart](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## III. Module API and Prediction


- ### 1、Command line Prediction

  - ```shell
    $ hub run chinese_ocr_db_crnn_mobile --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command line instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile", enable_mkldnn=True)       # MKLDNN acceleration is only available on CPU
    result = ocr.recognize_text(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    __init__(text_detector_module=None, enable_mkldnn=False)
    ```

    - Construct the ChineseOCRDBCRNN object

    - **Parameter**

      - text_detector_module(str): PaddleHub Module Name for text detection, use [chinese_text_detection_db_mobile Module](../chinese_text_detection_db_mobile/) by default if set to None. Its function is to detect the text in the picture.
      - enable_mkldnn(bool): Whether to enable MKLDNN to accelerate CPU computing. This parameter is valid only when the CPU is running. The default is False.


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

    - Prediction API, detecting the position of all Chinese text in the input image.

    - **Parameter**

      - paths (list\[str\]): image path
      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format \[H, W, C\], BGR;
      - use\_gpu (bool): use GPU or not **If GPU is used, set the CUDA_VISIBLE_DEVICES environment variable first**
      - box\_thresh (float): The confidence threshold of text box detection;
      - text\_thresh (float): The confidence threshold of Chinese text recognition;
      - angle_classification_thresh(float): The confidence threshold of text Angle classification
      - visualization (bool): Whether to save the recognition results as picture files;
      - output\_dir (str): path to save the image, ocr\_result by default.

    - **Return**

      - res (list\[dict\]): The list of recognition results, where each element is dict and each field is:
        - data (list\[dict\]): recognition result, each element in the list is dict and each field is:
          - text(str): The result text of recognition
          - confidence(float): The confidence of the results
          - text_box_position(list): The pixel coordinates of the text box in the original picture, a 4*2 matrix, represent the coordinates of the lower left, lower right, upper right and upper left vertices of the text box in turn
      data is \[\] if there's no result
        - save_path (str, optional): Path to save the result, save_path is '' if no image is saved.


## IV. Server Deployment

- PaddleHub Serving can deploy an online object detection service.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:
  - ```shell
    $ hub serving start -m chinese_ocr_db_crnn_mobile
    ```

  - The servitization API is now deployed and the default port number is 8866.

  - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before prediction. Otherwise, need not set it.


- ### Step 2: Send a predictive request

  - After configuring the server, the following lines of code can be used to send the prediction request and obtain the prediction result

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
    url = "http://127.0.0.1:8866/predict/chinese_ocr_db_crnn_mobile"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction result
    print(r.json()["results"])
    ```

## V. Release Note

* 1.0.0

  First release

* 1.0.1

  Fixed failure to use the online service invocating model

* 1.0.2

  Supports MKLDNN to speed up CPU computing

* 1.1.0

  An ultra-lightweight three-stage model (text box detection - angle classification - text recognition) is used to identify text in images.

* 1.1.1

   Supports recognition of spaces in text.

* 1.1.2

   Fixed an issue where only 30 fields can be detected.

* 1.1.3

   Remove fluid api

  - ```shell
    $ hub install chinese_ocr_db_crnn_mobile==1.1.3
    ```
