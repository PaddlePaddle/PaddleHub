# FCN_HRNet_W18_Face_Seg

|Module Name|FCN_HRNet_W18_Face_Seg|
| :--- | :---: |
|Category|image segmentation|
|Network|FCN_HRNet_W18|
|Dataset|-|
|Fine-tuning supported or not|No|
|Module Size|56MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/88155299a7534f1084f8467a4d6db7871dc4729627d3471c9129d316dc4ff9bc"  width='70%' hspace='10'/> <br />
    </p>


- ### Module Introduction

  - This module is based on FCN_HRNet_W18 model, and can be used to segment face region.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.0   | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)  

- ### 2、Installation

  - ```shell
    $ hub install FCN_HRNet_W18_Face_Seg
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="FCN_HRNet_W18_Face_Seg")
    result = model.Segmentation(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.Segmentation(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def Segmentation(images=None,
                    paths=None,
                    batch_size=1,
                    output_dir='output',
                    visualization=False):
    ```

    - Face segmentation API.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
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
    $ hub install FCN_HRNet_W18_Face_Seg==1.0.0
    ```
