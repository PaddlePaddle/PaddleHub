# Photo2Cartoon

|Module Name|Photo2Cartoon|
| :--- | :---: |
|Category|image generation|
|Network|U-GAT-IT|
|Dataset|cartoon_data|
|Fine-tuning supported or not|No|
|Module Size|205MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://img-blog.csdnimg.cn/20201224164040624.jpg"   hspace='10'/> <br />
    </p>



- ### Module Introduction

  - This module encapsulates project [photo2cartoon](https://github.com/minivision-ai/photo2cartoon-paddle).


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0   | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)  

- ### 2、Installation

  - ```shell
    $ hub install Photo2Cartoon
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="Photo2Cartoon")
    result = model.Cartoon_GEN(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.Cartoon_GEN(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def Cartoon_GEN(images=None,
                    paths=None,
                    batch_size=1,
                    output_dir='output',
                    visualization=False,
                    use_gpu=False):
    ```

    - Cartoon style generation API.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - output_dir (str): save path of images;
      - batch_size (int): the size of batch;
      - visualization (bool): Whether to save the results as picture files;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**
      - res (list\[numpy.ndarray\]): result list，ndarray.shape is \[H, W, C\]



## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install Photo2Cartoon==1.0.0
    ```
