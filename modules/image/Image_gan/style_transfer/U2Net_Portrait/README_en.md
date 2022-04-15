# U2Net_Portrait

|Module Name|U2Net_Portrait|
| :--- | :---: |
|Category|image generation|
|Network|U^2Net|
|Dataset|-|
|Fine-tuning supported or not|No|
|Module Size|254MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/07f73466f3294373965e06c141c4781992f447104a94471dadfabc1c3d920861"  height='50%' hspace='10'/>
    <br />
    Input image
    <br />
    <img src="https://ai-studio-static-online.cdn.bcebos.com/c6ab02cf27414a5ba5921d9e6b079b487f6cda6026dc4d6dbca8f0167ad7cae3"   height='50%' hspace='10'/>
    <br />
    Output image
    <br />
    </p>


- ### Module Introduction

  - U2Net_Portrait can be used to create a face portrait.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install U2Net_Portrait
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="U2Net_Portrait")
    result = model.Portrait_GEN(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.Portrait_GEN(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def Portrait_GEN(images=None,
                    paths=None,
                    scale=1,
                    batch_size=1,
                    output_dir='output',
                    face_detection=True,
                    visualization=False):
    ```

    - Portrait generation API.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - scale (float) : scale for resizing image；<br/>
      - batch_size (int): the size of batch;
      - output_dir (str): save path of images;
      - visualization (bool): Whether to save the results as picture files;

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**
      - res (list\[numpy.ndarray\]): result list，ndarray.shape is \[H, W, C\]



## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install U2Net_Portrait==1.0.0
    ```
