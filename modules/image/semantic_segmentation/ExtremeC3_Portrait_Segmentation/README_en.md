# ExtremeC3_Portrait_Segmentation

|Module Name|ExtremeC3_Portrait_Segmentation|
| :--- | :---: | 
|Category|image segmentation|
|Network |ExtremeC3|
|Dataset|EG1800, Baidu fashion dataset|
|Fine-tuning supported or not|No|
|Module Size|0.038MB|
|Data indicators|-|
|Latest update date|2021-02-26|

## I. Basic Information 

- ### Application Effect Display

    - Sample results:
        <p align="center">
        <img src="https://ai-studio-static-online.cdn.bcebos.com/1261398a98e24184852bdaff5a4e1dbd7739430f59fb47e8b84e3a2cfb976107"  hspace='10'/> <br />
        </p>


- ### Module Introduction
    * ExtremeC3_Portrait_Segmentation is a light weigth module based on ExtremeC3 to achieve portrait segmentation.

    * For more information, please refer to: [ExtremeC3_Portrait_Segmentation](https://github.com/clovaai/ext_portrait_segmentation).

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0  

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install ExtremeC3_Portrait_Segmentation
      ```
      
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_ch/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_ch/get_start/mac_quickstart.md) 


## III. Module API Prediction

- ### 1、Prediction Code Example

    ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='ExtremeC3_Portrait_Segmentation')

    result = model.Segmentation(
        images=[cv2.imread('/PATH/TO/IMAGE')],
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False)
    ```

- ### 2、API

    ```python
    def Segmentation(
        images=None,
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False):
    ```
    - Prediction API, used for portrait segmentation.

    - **Parameter**
        * images (list[np.ndarray]) : image data, ndarray.shape is in the format [H, W, C], BGR;
        * paths (list[str]) :image path
        * batch_size (int) : batch size
        * output_dir (str) : save path of images, 'output' by default.
        * visualization (bool) : whether to save the segmentation results as picture files.
    - **Return**
        * results (list[dict{"mask":np.ndarray,"result":np.ndarray}]): list of recognition results.

## IV. Release Note

- 1.0.0

  First release
