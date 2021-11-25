# stgan_celeba

|Module Name|stgan_celeba|
| :--- | :---: |
|Category|image generation|
|Network|STGAN|
|Dataset|Celeba|
|Fine-tuning supported or not|No|
|Module Size |287MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I. Basic Information 

- ### Application Effect Display
  - Sample results:

    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/137856070-2a43facd-cda0-473f-8935-e61f5dd583d8.JPG" width=1200><br/>
    The image attributes are: original image, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Gender, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Aged<br/>
    </p>


- ### Module Introduction

  - STGAN takes the original attribute and the target attribute as input, and proposes STUs (Selective transfer units) to select and modify features of the encoder. The PaddleHub Module is trained one Celeba dataset and currently supports attributes of "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Gender", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged".


## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 1.5.2 

  - paddlehub >= 1.0.0  | [How to install PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install stgan_celeba==1.0.0
    ```
  - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  
 

## III. Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run stgan_celeba --image "/PATH/TO/IMAGE" --info "original_attributes" --style "target_attribute" 
    ```
    - **Parameters**

      - image: Image path

      - info: Attributes of original image, must fill in gender（ "Male" or "Female").The options are "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged". For example, the input picture is a girl with black hair, then fill in as "Female,Black_Hair". 
    
      - style: Specify the attributes to be converted. The options are "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Gender", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged". You can choose one of the options.
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub

    stgan = hub.Module(name="stgan_celeba")

    test_img_path = ["/PATH/TO/IMAGE"]
    org_info = ["Female,Black_Hair"]
    trans_attr = ["Bangs"]

    # set input dict
    input_dict = {"image": test_img_path, "style": trans_attr, "info": org_info}

    # execute predict and print the result
    results = stgan.generate(data=input_dict)
    print(results)
    ```

- ### 3、API

  - ```python
    def generate(data)
    ```

    - Style transfer API.

    - **Parameter**

      - data(list[dict]): Each element in the list is dict and each field is: 
          - image (list\[str\])： Each element in the list is the path of the image to be converted.
          - style (list\[str\])： Each element in the list is a string, fill in the face attributes to be converted.
          - info (list\[str\])： Represents the face attributes of the original image. Different attributes are separated by commas.
          

    - **Return**
      - res (list\[str\]): Save path of the result.

## IV. Release Note

- 1.0.0

  First release
