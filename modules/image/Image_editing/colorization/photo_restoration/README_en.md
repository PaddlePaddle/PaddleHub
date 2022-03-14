# photo_restoration

|Module Name|photo_restoration|
| :--- | :---: | 
|Category|Image editing|
|Network|deoldify and realsr|
|Fine-tuning supported or not|No|
|Module Size |64MB+834MB|
|Data indicators|-|
|Latest update date|2021-08-19|



## I. Basic Information 

- ### Application Effect Display

  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130897828-d0c86b81-63d1-4e9a-8095-bc000b8c7ca8.jpg" width = "260" height = "400" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130897762-5c9fa711-62bc-4067-8d44-f8feff8c574c.png" width = "260" height = "400" hspace='10'/>
    </p>



- ### Module Introduction

    - Photo_restoration can restore old photos. It mainly consists of two parts: coloring and super-resolution. The coloring model is deoldify
     , and super resolution model is realsr. Therefore, when using this model, please install deoldify and realsr in advance.

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

    - NOTE: This Module relies on ffmpeg, Please install ffmpeg before using this Module.

      ```shell
      $ conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
      ```
   
- ### 2、Installation

    - ```shell
      $ hub install photo_restoration
      ```
      
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Prediction Code Example


  - ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='photo_restoration', visualization=True)
    im = cv2.imread('/PATH/TO/IMAGE')
    res = model.run_image(im)

    ```
- ### 2、API


  - ```python
    def run_image(self,
                  input,
                  model_select= ['Colorization', 'SuperResolution'],
                  save_path = 'photo_restoration'):
    ```

    - Predicition API,  produce repaired photos.

    - **Parameter**

        - input (numpy.ndarray｜str): Image data，numpy.ndarray or str. ndarray.shape is in the format [H, W, C], BGR.

        - model_select (list\[str\]): Mode selection，\['Colorization'\] only colorize the input image， \['SuperResolution'\] only increase the image resolution；
        default is \['Colorization', 'SuperResolution'\]。

        - save_path (str): Save path, default is 'photo_restoration'.

     - **Return**

        - output (numpy.ndarray): Restoration result，ndarray.shape is in the format [H, W, C], BGR.


## IV. Server Deployment

- PaddleHub Serving can deploy an online service of photo restoration.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

        - ```shell
          $ hub serving start -m photo_restoration
          ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

    - With a configured server, use the following lines of code to send the prediction request and obtain the result

      - ```python
        import requests
        import json
        import base64

        import cv2
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
        org_im = cv2.imread('PATH/TO/IMAGE')
        data = {'images':cv2_to_base64(org_im), 'model_select': ['Colorization', 'SuperResolution']}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/photo_restoration"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        img = base64_to_cv2(r.json()["results"])
        cv2.imwrite('PATH/TO/SAVE/IMAGE', img)
        ```


## V. Release Note

- 1.0.0

  First release

- 1.0.1

  Adapt to paddlehub2.0

