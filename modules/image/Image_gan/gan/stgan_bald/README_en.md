# stgan_bald

|Module Name|stgan_bald|
| :--- | :---: |
|Category|image generation|
|Network|STGAN|
|Dataset|CelebA|
|Fine-tuning supported or not|No|
|Module Size|287MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Please refer to this [link](https://aistudio.baidu.com/aistudio/projectdetail/1145381)

- ### Module Introduction

  - This module is based on STGAN model, trained on CelebA dataset, and can be used to predict bald appearance after 1, 3 and 5 years.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlehub >= 1.8.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install stgan_bald
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)
## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    stgan_bald = hub.Module(name="stgan_bald")
    result = stgan_bald.bald(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = stgan_bald.bald(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def bald(images=None,
             paths=None,
             use_gpu=False,
             visualization=False,
             output_dir="bald_output")
    ```

    - Bald appearance generation API.

    - **Parameters**
      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - visualization (bool): Whether to save the results as picture files;
      - output_dir (str): save path of images;

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**

      - res (list\[numpy.ndarray\]): result list，ndarray.shape is \[H, W, C\]

## IV.Server Deployment

- PaddleHub Serving can deploy an online service of bald appearance generation.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m stgan_bald
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
    import numpy as np


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    def base64_to_cv2(b64str):
      data = base64.b64decode(b64str.encode('utf8'))
      data = np.fromstring(data, np.uint8)
      data = cv2.imdecode(data, cv2.IMREAD_COLOR)
      return data

    # Send an HTTP request
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stgan_bald"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # save results
    one_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_0']), cv2.COLOR_RGB2BGR)
    three_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_1']), cv2.COLOR_RGB2BGR)
    five_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_2']), cv2.COLOR_RGB2BGR)
    cv2.imwrite("stgan_bald_server.png", one_year)
    ```


## V.Release Note

* 1.0.0

  First release
  - ```shell
    $ hub install stgan_bald==1.0.0
    ```
