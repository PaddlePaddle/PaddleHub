# ID_Photo_GEN

|Module Name |ID_Photo_GEN|
| :--- | :---: |
|Category|Image generation|
|Network|HRNet_W18|
|Dataset |-|
|Fine-tuning supported or not |No|
|Module Size|28KB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I. Basic Information 

- ### Application Effect Display
  - Sample results:
    <p align="center">
    <img src="https://img-blog.csdnimg.cn/20201224163307901.jpg" > 
    </p>


- ### Module Introduction

  - This model is based on face_landmark_localization and FCN_HRNet_W18_Face_Seg. It can generate ID photos with white, red and blue background


## II. Installation

- ### 1、Environmental Dependence 

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0

- ### 2、Installation

  - ```shell
    $ hub install ID_Photo_GEN
    ```

  - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  
 
 
## III. Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='ID_Photo_GEN')

    result = model.Photo_GEN(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=True,
    use_gpu=False)
    ```

- ### 2、API

  - ```python
    def Photo_GEN(
        images=None,
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False,
        use_gpu=False):
    ```

    - Prediction API, generating ID photos.

    - **Parameter**
        * images (list[np.ndarray]): Image data, ndarray.shape is in the format [H, W, C], BGR.
        * paths (list[str]): Image path
        * batch_size (int): Batch size
        * output_dir (str): Save path of images, output by default.
        * visualization (bool): Whether to save the recognition results as picture files.
        * use_gpu (bool): Use GPU or not. **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**

        **NOTE:** Choose one of `paths` and `images` to provide input data.

    - **Return**
    
      * results (list[dict{"write":np.ndarray,"blue":np.ndarray,"red":np.ndarray}]): The list of generation results.


## IV. Release Note

- 1.0.0

  First release