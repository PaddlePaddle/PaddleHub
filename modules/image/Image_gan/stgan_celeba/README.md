# stgan_celeba

|模型名称|stgan_celeba|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|STGAN|
|数据集|Celeba|
|是否支持Fine-tuning|否|
|模型大小|287MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：

    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/137856070-2a43facd-cda0-473f-8935-e61f5dd583d8.JPG" width=1200><br/>
    STGAN的效果图(图片属性分别为：original image, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Gender, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Aged)<br/>
    </p>


- ### 模型介绍

  - STGAN 以原属性和目标属性的差值作为输入，并创造性地提出了 STUs (Selective transfer units) 来选择和修改 encoder 的特征，从而改善了转换效果和处理能力。 该 PaddleHub Module 使用 Celeba 数据集训练完成，目前支持 "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Gender", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged" 这十三种人脸属性转换。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.5.2 

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install stgan_celeba==1.0.0
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
 

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run stgan_celeba --image "/PATH/TO/IMAGE" --info "original_attributes" --style "target_attribute" 
    ```
  - **参数**

    - image ：指定图片路径。

    - info ：原图的属性，必须填写性别（ "Male" 或者 "Female"）。可选值有："Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged" 。比如输入图片是一个女孩，有着黑头发，那么就填写为 "Female,Black_Hair"。建议尽可能完整地填写原图具备的属性，比如一个黑发女孩还戴了眼镜，那么应填写为 "Female,Black_Hair,Eyeglasses"，否则有可能转换失败。
    
    - style 指定拟转换的属性，可选择 "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Gender", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged" 中的一种。

- ### 2、预测代码示例

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

    - 风格转换API，用于图像生成。

    - **参数**

      - data： dict 类型，有以下字段
          - image (list\[str\])： list中每个元素为待转换的图片路径。
          - style (list\[str\])： list中每个元素为字符串，填写待转换的人脸属性。
          - info (list\[str\])： 表示原图具备的人脸属性，填得越详细效果会越好，不同属性用逗号隔开。
          

    - **返回**
      - res (list\[str\]): 提示生成图片的保存路径。



## 四、更新历史

* 1.0.0

  初始发布
