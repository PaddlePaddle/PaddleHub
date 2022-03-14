# stargan_celeba

|Module Name|stargan_celeba|
| :--- | :---: |
|Category|image generation|
|Network|STGAN|
|Dataset|Celeba|
|Fine-tuning supported or not|No|
|Module Size |33MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I. Basic Information 

- ### Application Effect Display
  - Sample results:

    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/137855887-f0abca76-2735-4275-b7ad-242decf31bb3.PNG" width=600><br/>
     The image attributes are: origial image, Black_Hair, Blond_Hair, Brown_Hair, Male, Aged<br/>
    </p>


- ### Module Introduction

  - STGAN takes the original attribute and the target attribute as input, and  proposes STUs (Selective transfer units) to select and modify features of the encoder. The PaddleHub Module is trained one Celeba dataset and currently supports attributes of  "Black_Hair", "Blond_Hair", "Brown_Hair", "Female", "Male", "Aged".


## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 1.5.2 

  - paddlehub >= 1.0.0  | [How to install PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install stargan_celeba==1.0.0
    ```
  - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  
 

## III. Module API Prediction

- ### 1、Command line Prediction

    - ```shell
      $ hub run stargan_celeba --image "/PATH/TO/IMAGE" --style "target_attribute"
      ```

    - **Parameters**

      - image: image path

      - style: Specify the attributes to be converted. The options are "Black_Hair", "Blond_Hair", "Brown_Hair", "Female", "Male", "Aged". You can choose one of the options.

    - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub

    stargan = hub.Module(name="stargan_celeba")
    test_img_path = ["/PATH/TO/IMAGE"]
    trans_attr = ["Blond_Hair"]

    # set input dict
    input_dict = {"image": test_img_path, "style": trans_attr}

    # execute predict and print the result
    results = stargan.generate(data=input_dict)
    print(results)
    ```

- ### 3、API

  - ```python
    def generate(data)
    ```

    - Style transfer API.

    - **Parameter**

      - data(list[dict]): each element in the list is dict and each field is: 
          - image (list\[str\])： Each element in the list is the path of the image to be converted.
          - style (list\[str\])： Each element in the list is a string, fill in the face attributes to be converted.

    - **Return**
      - res (list\[str\]): Save path of the result.

## IV. Release Note

- 1.0.0

  First release
