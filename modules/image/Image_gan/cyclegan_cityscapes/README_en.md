# cyclegan_cityscapes

|Module Name|cyclegan_cityscapes|
| :--- | :---: |
|Category |Image generation|
|Network |CycleGAN|
|Dataset|Cityscapes|
|Fine-tuning supported or not |No|
|Module Size |33MB|
|Latest update date |2021-02-26|
|Data indicators|-|


## I. Basic Information 


- ### Application Effect Display

  - Sample results:

    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/137839740-4be4cf40-816f-401e-a73f-6cda037041dd.png"  width = "450" height = "300" hspace='10'/>
     <br />
    Input image
     <br />
    <img src="https://user-images.githubusercontent.com/35907364/137839777-89fc705b-f0d7-4a93-94e2-76c0d3c5a0b0.png"  width = "450" height = "300" hspace='10'/>
     <br />
    Output image
     <br />
    </p>


- ### Module Introduction

  - CycleGAN belongs to Generative Adversarial Networks(GANs). Unlike traditional GANs that can only generate pictures in one direction, CycleGAN can simultaneously complete the style transfer of two domains. The PaddleHub Module is trained by Cityscapes dataset, and supports the conversion from real images to semantic segmentation results, and also supports conversion from semantic segmentation results to real images.


## II. Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.4.0

  - paddlehub >= 1.1.0 

- ### 2、Installation

  - ```shell
    $ hub install cyclegan_cityscapes==1.0.0
    ```
  - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  

 
## III. Module API Prediction

- ### 1、Command line Prediction
  - ```shell
    $ hub run cyclegan_cityscapes --input_path "/PATH/TO/IMAGE"
    ```
  
    - **Parameters**

      - input_path: image path
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub

    cyclegan = hub.Module(name="cyclegan_cityscapes")

    test_img_path = "/PATH/TO/IMAGE"

    # set input dict
    input_dict = {"image": [test_img_path]}

    # execute predict and print the result
    results = cyclegan.generate(data=input_dict)
    print(results)
    ```

- ### 3、API

  - ```python
    def generate(data)
    ```

    - Style transfer API.

    - **Parameters**

      - data(list[dict]): Each element in the list is dict and each field is:
          - image (list\[str\])： Image path.

    - **Return**
      - res (list\[str\]): The list of style transfer results, where each element is dict and each field is: 
          - origin: Original input path.
          - generated: Save path of images.



## IV. Release Note

* 1.0.0

  First release

