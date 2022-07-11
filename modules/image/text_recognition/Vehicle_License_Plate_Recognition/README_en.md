# Vehicle_License_Plate_Recognition

|Module Name|Vehicle_License_Plate_Recognition|
| :--- | :---: |
|Category|text recognition|
|Network|-|
|Dataset|CCPD|
|Fine-tuning supported or not|No|
|Module Size|111MB|
|Latest update date|2021-03-22|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/35a3dab32ac948549de41afba7b51a5770d3f872d60b437d891f359a5cef8052"  width = "450" height = "300" hspace='10'/> <br />
    </p>


- ### Module Introduction

  - Vehicle_License_Plate_Recognition is a module for licence plate recognition, trained on CCPD dataset. This model can detect the position of licence plate and recognize the contents.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.4

  - paddleocr >= 2.0.2  

- ### 2、Installation

  - ```shell
    $ hub install Vehicle_License_Plate_Recognition
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="Vehicle_License_Plate_Recognition")
    result = model.plate_recognition(images=[cv2.imread('/PATH/TO/IMAGE')])
    ```

- ### 2、API

  - ```python
    def plate_recognition(images)
    ```

    - Prediction API.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;


    - **Return**
      - results(list(dict{'license', 'bbox'})): The list of recognition results, where each element is dict.


## IV.Server Deployment

- PaddleHub Serving can deploy an online service of text recognition.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m Vehicle_License_Plate_Recognition
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
    url = "http://127.0.0.1:8866/predict/Vehicle_License_Plate_Recognition"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install Vehicle_License_Plate_Recognition==1.0.0
    ```
