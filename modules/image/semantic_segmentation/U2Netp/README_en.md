# U2Netp

|Module Name |U2Netp|
| :--- | :---: |
|Category |Image segmentation|
|Network |U^2Net|
|Dataset|-|
|Fine-tuning supported or not|No|
|Module Size |6.7MB|
|Data indicators|-|
|Latest update date|2021-02-26|


## I. Basic Information 

- ### Application Effect Display

  - Sample results:

        <p align="center">
        <img src="https://ai-studio-static-online.cdn.bcebos.com/4d77bc3a05cf48bba6f67b797978f4cdf10f38288b9645d59393dd85cef58eff" width = "450" height = "300" hspace='10'/> <img src="https://ai-studio-static-online.cdn.bcebos.com/11c9eba8de6d4316b672f10b285245061821f0a744e441f3b80c223881256ca0" width = "450" height = "300" hspace='10'/>
        </p>


- ### Module Introduction

    - Network architecture:
      <p align="center">
      <img src="https://ai-studio-static-online.cdn.bcebos.com/999d37b4ffdd49dc9e3315b7cec7b2c6918fdd57c8594ced9dded758a497913d" hspace='10'/> <br />
      </p>

    - For more information, please refer to: [U2Net](https://github.com/xuebinqin/U-2-Net)


## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0  
    - paddlehub >= 2.0.0

- ### 2、Installation
    - ```shell
      $ hub install U2Netp
      ```

    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md) 

## III. Module API Prediction

- ### 1、Prediction Code Example

    ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='U2Netp')

    result = model.Segmentation(
        images=[cv2.imread('/PATH/TO/IMAGE')],
        paths=None,
        batch_size=1,
        input_size=320,
        output_dir='output',
        visualization=True)
    ```
 - ### 2、API

    ```python
    def Segmentation(
            images=None,
            paths=None,
            batch_size=1,
            input_size=320,
            output_dir='output',
            visualization=False):
    ```
    - Prediction API, obtaining segmentation result.

    - **Parameter**
        * images (list[np.ndarray]) : Image data, ndarray.shape is in the format [H, W, C], BGR.
        * paths (list[str]) : Image path.
        * batch_size (int) : Batch size.
        * input_size (int) : Input image size, default is 320.
        * output_dir (str) : Save path of images, 'output' by default.
        * visualization (bool) : Whether to save the results as picture files.

    - **Return**
        * results (list[np.ndarray]): The list of segmentation results.

## IV. Release Note

- 1.0.0

  First release
