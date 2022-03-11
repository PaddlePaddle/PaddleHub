# animegan_v2_paprika_54

|Module Name |animegan_v2_paprika_54|
| :--- | :---: |
|Category |Image generation|
|Network|AnimeGAN|
|Dataset|Paprika|
|Fine-tuning supported or not|No|
|Module Size|9.4MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I. Basic Information 

- ### Application Effect Display

  - Sample results:
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/bd002c4bb6a7427daf26988770bb18648b7d8d2bfd6746bfb9a429db4867727f"  width = "450" height = "300" hspace='10'/>
    <br />
    Input image
    <br />
    <img src="https://ai-studio-static-online.cdn.bcebos.com/6574669d87b24bab9627c6e33896528b4a0bf5af1cd84ca29655d68719f2d551"  width = "450" height = "300" hspace='10'/>
    <br />
    Output image
     <br />
    </p>



- ### Module Introduction

  - AnimeGAN V2 image style stransfer model, the model can convert the input image into red pepper anime style, the model weight is converted from[AnimeGAN V2 official repo](https://github.com/TachibanaYoshino/AnimeGAN)。


## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 1.8.0  

  - paddlehub >= 1.8.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install animegan_v2_paprika_54
    ```

  - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="animegan_v2_paprika_54")
    result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def style_transfer(images=None,
                       paths=None,
                       output_dir='output',
                       visualization=False,
                       min_size=32,
                       max_size=1024)
    ```

    - Style transfer API.

    - **Parameters**

      - images (list\[numpy.ndarray\]): Image data, ndarray.shape is in the format [H, W, C], BGR.
      - paths (list\[str\]): Image path.
      - output\_dir (str): Save path of images, `output` by default.
      - visualization (bool): Whether to save the results as picture files.
      - min\_size (int): Minimum size, default is  32.
      - max\_size (int): Maximum size, default is 1024.

      **NOTE:** Choose one of `paths` and `images` to provide input data.

    - **Return**
      - res (list\[numpy.ndarray\]): The list of style transfer results，ndarray.shape is in the format [H, W, C].


## IV. Server Deployment

- PaddleHub Serving can deploy an online service of style transfer.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:

    - ```shell
      $ hub serving start -m animegan_v2_paprika_54
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
        url = "http://127.0.0.1:8866/predict/animegan_v2_paprika_54"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # print prediction results
        print(r.json()["results"])
        ```


## V. Release Note

- 1.0.0

  First release.

* 1.0.1

  Support paddlehub2.0.

* 1.0.2

  Delete batch_size.
