# attgan_celeba

|模型名称|attgan_celeba|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|AttGAN|
|数据集|Celeba|
|是否支持Fine-tuning|否|
|模型大小|167MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：

    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/137855667-43c5c40c-28f5-45d8-accc-028e185b988f.JPG" width=1200><br/>
    图1. AttGAN的效果图(图片属性分别为：original image, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Gender, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Aged)<br/>
    </p>


- ### 模型介绍

  - AttGAN 是一种生成对抗网络(Generative Adversarial Networks)，它利用分类损失和重构损失来保证改变特定的属性。该 PaddleHub Module 使用 Celeba 数据集训练完成，目前支持 "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Gender", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged" 这十三种人脸属性转换。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.5.2 

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install attgan_celeba==1.0.0
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
 

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run attgan_celeba --image "/PATH/TO/IMAGE" --style "target_attribute" 
    ```
  - **参数**

    - image ：指定图片路径。

    - style 指定拟转换的属性，可选择 "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Bushy_Eyebrows", "Eyeglasses", "Gender", "Mouth_Slightly_Open", "Mustache", "No_Beard", "Pale_Skin", "Aged" 中的一种。



- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    attgan = hub.Module(name="attgan_celeba")

    test_img_path = ["/PATH/TO/IMAGE"]
    trans_attr = ["Bangs"]

    # set input dict
    input_dict = {"image": test_img_path, "style": trans_attr}

    # execute predict and print the result
    results = attgan.generate(data=input_dict)
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

    - **返回**
      - res (list\[str\]): 提示生成图片的保存路径。



## 四、更新历史

* 1.0.0

  初始发布


