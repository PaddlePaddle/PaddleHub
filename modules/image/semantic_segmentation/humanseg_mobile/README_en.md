# humanseg_mobile

|Module Name |humanseg_mobile|
| :--- | :---: |
|Category |Image segmentation|
|Network|hrnet|
|Dataset|Baidu self-built dataset|
|Fine-tuning supported or not|No|
|Module Size|5.8M|
|Data indicators|-|
|Latest update date|2021-02-26|

## I. Basic Information

- ### Application Effect Display

  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130913092-312a5f37-842e-4fd0-8db4-5f853fd8419f.jpg" width = "337" height = "505" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130914325-3795e241-b611-46a1-aa70-ffc47326c86a.png" width = "337" height = "505" hspace='10'/>
    </p>

- ### Module Introduction

    - HumanSeg_mobile is based on HRNet_w18_small_v1 network. The network size is only 5.8M. It is suitable for selfie portrait segmentation and can be segmented in real time on the mobile terminal.

    - For more information, please refer to:[humanseg_mobile](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/HumanSeg)


## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install humanseg_mobile
      ```

    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  

## III. Module API Prediction

- ### 1、Command line Prediction

    - ```
      hub run humanseg_mobile --input_path "/PATH/TO/IMAGE"

      ```
    - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)


- ### 2、Prediction Code Example
    - Image segmentation and video segmentation example：
        ```python
        import cv2
        import paddlehub as hub

        human_seg = hub.Module(name='humanseg_mobile')
        im = cv2.imread('/PATH/TO/IMAGE')
        res = human_seg.segment(images=[im],visualization=True)
        print(res[0]['data'])
        human_seg.video_segment('/PATH/TO/VIDEO')
        ```
    - Video prediction example:

        ```python
        import cv2
        import numpy as np
        import paddlehub as hub

        human_seg = hub.Module('humanseg_mobile')
        cap_video = cv2.VideoCapture('\PATH\TO\VIDEO')
        fps = cap_video.get(cv2.CAP_PROP_FPS)
        save_path = 'humanseg_mobile_video.avi'
        width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
        prev_gray = None
        prev_cfd = None
        while cap_video.isOpened():
            ret, frame_org = cap_video.read()
            if ret:
                [img_matting, prev_gray, prev_cfd] = human_seg.video_stream_segment(frame_org=frame_org, frame_id=cap_video.get(1), prev_gray=prev_gray, prev_cfd=prev_cfd)
                img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)
                bg_im = np.ones_like(img_matting) * 255
                comb = (img_matting * frame_org + (1 - img_matting) * bg_im).astype(np.uint8)
                cap_out.write(comb)
            else:
                break

        cap_video.release()
        cap_out.release()

        ```

- ### 3、API

    ```python
    def segment(images=None,
                paths=None,
                batch_size=1,
                use_gpu=False,
                visualization=False,
                output_dir='humanseg_mobile_output')
    ```

    - Prediction API, generating segmentation result.

    - **Parameter**

        * images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR.
        * paths (list\[str\]): image path.
        * batch\_size (int): batch size.
        * use\_gpu (bool): use GPU or not. **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
        * visualization (bool): Whether to save the results as picture files.
        * output\_dir (str): save path of images, humanseg_mobile_output by default.

    - **Return**

        * res (list\[dict\]): The list of recognition results, where each element is dict and each field is:
            * save\_path (str, optional): Save path of the result.
            * data (numpy.ndarray): The result of portrait segmentation.

    ```python
    def video_stream_segment(self,
                            frame_org,
                            frame_id,
                            prev_gray,
                            prev_cfd,
                            use_gpu=False):
    ```

    -  Prediction API, used to segment video portraits frame by frame.

    - **Parameter**

        * frame_org (numpy.ndarray): single frame for prediction，ndarray.shape is in the format [H, W, C], BGR.
        * frame_id (int): The number of the current frame.
        * prev_gray (numpy.ndarray): Grayscale image of the previous network input.
        * prev_cfd (numpy.ndarray): The fusion image from optical flow and the prediction result from previous frame.
        * use\_gpu (bool): Use GPU or not. **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**


    - **Return**

        * img_matting (numpy.ndarray): The result of portrait segmentation.
        * cur_gray (numpy.ndarray): Grayscale image of the current network input.
        * optflow_map (numpy.ndarray): The fusion image from optical flow and the prediction result from current frame.


    ```python
    def video_segment(self,
                      video_path=None,
                      use_gpu=False,
                      save_dir='humanseg_mobile_video_result'):
    ```

    -  Prediction API to produce video segmentation result.

    - **Parameter**

        * video\_path (str): Video path for segmentation。If None, the video will be obtained from the local camera, and a window will display the online segmentation result.
        * use\_gpu (bool): Use GPU or not. **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
        * save\_dir (str): save path of video.


    ```python
    def save_inference_model(dirname)
    ```


    - Save the model to the specified path.

    - **Parameters**

      * dirname: Model save path.



## IV. Server Deployment

- PaddleHub Serving can deploy an online service of for human segmentation.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

        -  ```shell
           $ hub serving start -m humanseg_mobile
           ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

    - With a configured server, use the following lines of code to send the prediction request and obtain the result

        ```python
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
        org_im = cv2.imread('/PATH/TO/IMAGE')
        data = {'images':[cv2_to_base64(org_im)]}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/humanseg_mobile"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        mask =cv2.cvtColor(base64_to_cv2(r.json()["results"][0]['data']), cv2.COLOR_BGR2GRAY)
        rgba = np.concatenate((org_im, np.expand_dims(mask, axis=2)), axis=2)
        cv2.imwrite("segment_human_mobile.png", rgba)
        ```

- ### Gradio APP support
   Starting with PaddleHub 2.3.1, the Gradio APP for humanseg_mobile is supported to be accessed in the browser using the link http://127.0.0.1:8866/gradio/humanseg_mobile.

## V. Release Note

- 1.0.0

    First release

- 1.1.0

    Added video portrait split interface

    Added video stream portrait segmentation interface

* 1.1.1

    Fix the video memory leakage problem of on cudnn 8.0.4

* 1.2.0

    Remove Fluid API

* 1.3.0

    Add Gradio APP support.

    ```shell
    $ hub install humanseg_mobile == 1.3.0
    ```
