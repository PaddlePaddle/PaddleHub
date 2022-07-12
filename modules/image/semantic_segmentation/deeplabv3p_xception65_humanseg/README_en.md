# deeplabv3p_xception65_humanseg

|Module Name |deeplabv3p_xception65_humanseg|
| :--- | :---: |
|Category|Image segmentation|
|Network|deeplabv3p|
|Dataset|Baidu self-built dataset|
|Fine-tuning supported or not|No|
|Module Size|162MB|
|Data indicators |-|
|Latest update date|2021-02-26|

## I. Basic Information

- ### Application Effect Display

  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130913092-312a5f37-842e-4fd0-8db4-5f853fd8419f.jpg" width = "337" height = "505" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130913256-41056b21-1c3d-4ee2-b481-969c94754609.png" width = "337" height = "505" hspace='10'/>
    </p>

- ### Module Introduction

  - DeepLabv3+ model is trained by Baidu self-built dataset, which can be used for portrait segmentation.
<p align="center">
<img src="https://paddlehub.bj.bcebos.com/paddlehub-img/deeplabv3plus.png" hspace='10'/> <br />
</p>

- For more information, please refer to: [deeplabv3p](https://github.com/PaddlePaddle/PaddleSeg)

## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install deeplabv3p_xception65_humanseg
      ```
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    hub run deeplabv3p_xception65_humanseg --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)



- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
    result = human_seg.segmentation(images=[cv2.imread('/PATH/TO/IMAGE')])
    ```

- ### 3.API

    - ```python
      def segmentation(images=None,
                      paths=None,
                      batch_size=1,
                      use_gpu=False,
                      visualization=False,
                      output_dir='humanseg_output')
      ```

      - Prediction API, generating segmentation result.

      - **Parameter**
        * images (list\[numpy.ndarray\]): Image data, ndarray.shape is in the format [H, W, C], BGR.
        * paths (list\[str\]): Image path.
        * batch\_size (int): Batch size.
        * use\_gpu (bool): Use GPU or not. **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
        * visualization (bool): Whether to save the recognition results as picture files.
        * output\_dir (str): Save path of images.

      - **Return**

          * res (list\[dict\]): The list of recognition results, where each element is dict and each field is:
              * save\_path (str, optional): Save path of the result.
              * data (numpy.ndarray): The result of portrait segmentation.

    - ```python
      def save_inference_model(dirname,
                              model_filename=None,
                              params_filename=None,
                              combined=True)
      ```

      - Save the model to the specified path.

      - **Parameters**
        * dirname: Save path.
        * model\_filename: Model file name，defalt is \_\_model\_\_
        * params\_filename: Parameter file name，defalt is \_\_params\_\_(Only takes effect when `combined` is True)
        * combined: Whether to save the parameters to a unified file.


## IV. Server Deployment

- PaddleHub Serving can deploy an online service of for human segmentation.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

      - ```shell
        $ hub serving start -m deeplabv3p_xception65_humanseg
        ```

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

    org_im = cv2.imread("/PATH/TO/IMAGE")
    # Send an HTTP request
    data = {'images':[cv2_to_base64(org_im)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/deeplabv3p_xception65_humanseg"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    mask =cv2.cvtColor(base64_to_cv2(r.json()["results"][0]['data']), cv2.COLOR_BGR2GRAY)
    rgba = np.concatenate((org_im, np.expand_dims(mask, axis=2)), axis=2)
    cv2.imwrite("segment_human_server.png", rgba)
    ```
## V. Release Note

- 1.0.0

  First release

* 1.1.0

   Improve prediction performance

* 1.1.1

   Fix the bug of image value out of range

* 1.1.2

   Remove fluid api

  - ```shell
    $ hub install deeplabv3p_xception65_humanseg==1.1.2
    ```
