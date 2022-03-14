# MiDaS_Large

|Module Name|MiDaS_Large|
| :--- | :---: |
|Category|depth estimation|
|Network|-|
|Dataset|3D Movies, WSVD, ReDWeb, MegaDepth|
|Fine-tuning supported or not|No|
|Module Size|399MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://img-blog.csdnimg.cn/20201227112600975.jpg"  width='70%' hspace='10'/> <br />
    </p>


- ### Module Introduction

  - MiDas_Large module is used for monocular depth estimation.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0   | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)  

- ### 2、Installation

  - ```shell
    $ hub install MiDaS_Large
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="MiDaS_Large")
    result = model.depth_estimation(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.depth_estimation(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def depth_estimation(images=None,
                    paths=None,
                    batch_size=1,
                    output_dir='output',
                    visualization=False):
    ```

    - depth estimation API.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - batch_size (int): the size of batch;
      - output_dir (str): save path of images;
      - visualization (bool): Whether to save the results as picture files;

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**
      - res (list\[numpy.ndarray\]): depth data，ndarray.shape is  \[H, W\]


## IV.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install MiDaS_Large==1.0.0
    ```
