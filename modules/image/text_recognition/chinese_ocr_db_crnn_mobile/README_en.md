# chinese_ocr_db_crnn_mobile

| Module Name                  | chinese_ocr_db_crnn_mobile       |
| ---------------------------- | -------------------------------- |
| Category                     | image-text recognition           |
| Network                      | Differentiable Binarization+RCNN |
| Dataset                      | icdar2015                        |
| Fine-tuning supported or not | No                               |
| Module Size                  | 16M                              |
| Latest update date           | 2021-02-26                       |
| Data indicators              | -                                |

## I. Basic Information

- ### Application Effect Display

  - [Online experience of OCR text recognition scenarios](https://www.paddlepaddle.org.cn/hub/scene/ocr)
  - Sample results:

[![img](https://user-images.githubusercontent.com/76040149/133097562-d8c9abd1-6c70-4d93-809f-fa4735764836.png)](https://user-images.githubusercontent.com/76040149/133097562-d8c9abd1-6c70-4d93-809f-fa4735764836.png)

- ### Module Introduction

  - chinese_ocr_db_crnn_mobile Module is used to identify Chinese characters in pictures. It first obtains the text box detected by [chinese_text_detection_db_mobile Module](), then identifies the Chinese characters and carries out angle classification to these text boxes. CRNN(Convolutional Recurrent Neural Network) is adopted as the final recognition algorithm. This Module is an ultra-lightweight Chinese OCR model that supports direct prediction.

[![img](https://user-images.githubusercontent.com/76040149/133098254-7c642826-d6d7-4dd0-986e-371622337867.png)](https://user-images.githubusercontent.com/76040149/133098254-7c642826-d6d7-4dd0-986e-371622337867.png)

- For more information, please refer to:[An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)

## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 1.7.2

  - paddlehub >= 1.6.0   | [How to install PaddleHub]()

  - shapely

  - pyclipper

  - ```
    $ pip install shapely pyclipper
    ```

  - **This Module relies on the third-party libraries, shapely and pyclipper. Please install shapely and pyclipper before using this Module.**

- ### 2、Installation

  - ```
    $ hub install chinese_ocr_db_crnn_mobile
    ```

  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III. Module API Prediction

- ### 1、Command line Prediction

  - ```
    $ hub run chinese_ocr_db_crnn_mobile --input_path "/PATH/TO/IMAGE"
    ```

  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction]()

- ### 2、Prediction Code Example

  - ```
    import paddlehub as hub
    import cv2

    ocr = hub.Module(name="chinese_ocr_db_crnn_mobile", enable_mkldnn=True)       # MKLDNN acceleration is only available on CPU
    result = ocr.recognize_text(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```
    __init__(text_detector_module=None, enable_mkldnn=False)
    ```

    - Construct the Chinese OCRDBCRNN object
    - **Parameters**
      - text_detector_module(str): Name of text detection module in PaddleHub Module, if set to None, [chinese_text_detection_db_mobile Module]() will be used by default. It serves to detect the text in the picture.
      - enable_mkldnn(bool): Whether to enable MKLDNN for CPU computing acceleration. This parameter is valid only when the CPU is running. The default is False.

  - ```
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
      - paths (list[str]): image path
      - images (list[numpy.ndarray]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - box_thresh (float): The confidence threshold for text box detection;
      - text_thresh (float): The confidence threshold for Chinese text recognition;
      - angle_classification_thresh(float): The confidence threshold for text angle classification
      - visualization (bool): Whether to save the recognition results as picture files;
      - output_dir (str): save path of images, ocr_result by default.
    - **Return**
      - res (list[dict]): The list of recognition results, where each element is dict and each field is:
        - data (list[dict]): recognition results, each element in the list is dict and each field is:
          - text(str): Recognized texts
          - confidence(float): The confidence of the results
          - text_box_position(list): The pixel coordinates of the text box in the original picture, a 4*2 matrix representing the coordinates of the lower left, lower right, upper right and upper left vertices of the text box in turn, data is [] if there's no result
        - save_path (str, optional): Save path of the result, save_path is '' if no image is saved.

## IV. Server Deployment

- PaddleHub Serving can deploy an online service of object detection.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:

  - ```
    $ hub serving start -m chinese_ocr_db_crnn_mobile
    ```

  - The servitization API is now deployed and the default port number is 8866.

  - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

  - With a configured server, use the following lines of code to send the prediction request and obtain the result

  - ```
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

    # print prediction results
    print(r.json()["results"])
    ```

## V. Release Note

- 1.0.0

  First release

- 1.0.1

  Fix the problem of model call failure using online service

- 1.0.2

  Support CPU computing acceleration based on MKLDNN

- 1.1.0

  Adopt an ultra-lightweight three-stage model (text box detection - angle classification - text recognition) to identify texts in images.

- 1.1.1

  Support recognition of spaces in texts.

- 1.1.2

  Fix an issue that only 30 fields can be detected.

  - ```
    $ hub install chinese_ocr_db_crnn_mobile==1.1.2
    ```
