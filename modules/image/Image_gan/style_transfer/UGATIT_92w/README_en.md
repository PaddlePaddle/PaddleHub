# UGATIT_92w

|Module Name|UGATIT_92w|
| :--- | :---: |
|Category|Image editing|
|Network |U-GAT-IT|
|Dataset|selfie2anime|
|Fine-tuning supported or not|No|
|Module Size|41MB|
|Latest update date |2021-02-26|
|Data indicators|-|


## I. Basic Information 

- ### Application Effect Display
  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/136651638-33cac040-edad-41ac-a9ce-7c0e678d8c52.jpg" width = "400" height = "400" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/136653047-f00c30fb-521f-486f-8247-8d8f63649473.jpg" width = "400" height = "400" hspace='10'/>
    </p>



- ### Module Introduction

  - UGATIT  can transfer the input face image into the anime style.


## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 1.8.2  

  - paddlehub >= 1.8.0

- ### 2、Installation

  - ```shell
    $ hub install UGATIT_92w
    ```

    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  
 
## III. Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='UGATIT_92w', use_gpu=False)
    result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def style_transfer(
        self,
        images=None,
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False
    )
    ```

    - Style transfer API, convert the input face image into anime style.

    - **Parameters**
        * images (list\[numpy.ndarray\]): Image data, ndarray.shape is in the format [H, W, C], BGR.
        * paths (list\[str\]): Image path，default is None；
        * batch\_size (int): Batch size, default is 1；
        * visualization (bool): Whether to save the recognition results as picture files, default is False.
        * output\_dir (str): save path of images, `output` by default.

      **NOTE:** Choose one of `paths` and `images` to provide input data.

    - **Return**

      - res (list\[numpy.ndarray\]): Style tranfer result,  ndarray.shape is in the format [H, W, C].

## IV. Server Deployment

- PaddleHub Serving can deploy an online service of Style transfer task.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:
  
    - ```shell
      $ hub serving start -m UGATIT_92w
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
      url = "http://127.0.0.1:8866/predict/UGATIT_92w"
      r = requests.post(url=url, headers=headers, data=json.dumps(data))

      # print prediction results
      print(r.json()["results"])
      ```

## V. Release Note

- 1.0.0

  First release