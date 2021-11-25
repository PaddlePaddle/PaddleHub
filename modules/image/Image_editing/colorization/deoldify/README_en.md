# deoldify

| Module Name |deoldify|
| :--- | :---: | 
|Category|Image editing|
|Network |NoGAN|
|Dataset|ILSVRC 2012|
|Fine-tuning supported or not |No|
|Module Size |834MB|
|Data indicators|-|
|Latest update date |2021-04-13|


## I. Basic Information 

- ### Application Effect Display
  
  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130886749-668dfa38-42ed-4a09-8d4a-b18af0475375.jpg" width = "450" height = "300" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130886685-76221736-839a-46a2-8415-e5e0dd3b345e.png" width = "450" height = "300" hspace='10'/>
    </p>

- ### Module Introduction

  - Deoldify is a color rendering model for images and videos, which can restore color for black and white photos and videos.

  - For more information, please refer to: [deoldify](https://github.com/jantic/DeOldify)

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
      $ hub install deoldify
      ```
      
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

  - ### 1、Prediction Code Example

    - ```python
      import paddlehub as hub

      model = hub.Module(name='deoldify')
      model.predict('/PATH/TO/IMAGE/OR/VIDEO')
      ```

  - ### 2、API

    - ```python
        def predict(self, input):
        ```

        - Prediction API.

        - **Parameter**

            - input (str): Image path.

        - **Return**

            - If input is image path, the output is：
              - pred_img(np.ndarray): image data, ndarray.shape is in the format [H, W, C], BGR.
              - out_path(str): save path of images.

            - If input is video path, the output is ：
              - frame_pattern_combined(str): save path of frames from output video.
              - vid_out_path(str): save path of output video.

    - ```python
      def run_image(self, img):
      ```
        - Prediction API for image.

        - **Parameter**

            - img (str｜np.ndarray): Image data,  str or ndarray. ndarray.shape is in the format [H, W, C], BGR.

        - **Return**

            - pred_img(np.ndarray): Ndarray.shape is in the format [H, W, C], BGR.

    - ```python
      def run_video(self, video):
      ```
       -  Prediction API for video.

       - **Parameter**

         - video(str): Video path.

       - **Return**

         - frame_pattern_combined(str): Save path of frames from output video.
         - vid_out_path(str): Save path of output video.


## IV. Server Deployment

- PaddleHub Serving can deploy an online service of coloring old photos or videos.


- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

      - ```shell
        $ hub serving start -m deoldify
        ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

    - With a configured server, use the following lines of code to send the prediction request and obtain the result.

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
        org_im = cv2.imread('/PATH/TO/ORIGIN/IMAGE')
        data = {'images':cv2_to_base64(org_im)}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/deoldify"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        img = base64_to_cv2(r.json()["results"])
        cv2.imwrite('/PATH/TO/SAVE/IMAGE', img)
        ```


## V. Release Note

- 1.0.0

  First release

- 1.0.1

  Adapt to paddlehub2.0
