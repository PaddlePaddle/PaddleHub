# stargan_celeba

|模型名称|stargan_celeba|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|STGAN|
|数据集|Celeba|
|是否支持Fine-tuning|否|
|模型大小|33MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：

    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/137855887-f0abca76-2735-4275-b7ad-242decf31bb3.PNG" width=600><br/>
    图1. StarGAN的效果图 (属性分别为：origial image, Black_Hair, Blond_Hair, Brown_Hair, Male, Aged)<br/>
    </p>


- ### 模型介绍

  - StarGAN 是为了解决跨多个域、多个数据集的训练而提出的生成对抗网络模型。单个 StarGAN 模型就可以实现多个风格域的转换。 该 PaddleHub Module 使用 Celeba 数据集训练完成，目前支持 "Black_Hair", "Blond_Hair", "Brown_Hair", "Female", "Male", "Aged" 这六种人脸属性转换。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.5.2 

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install stargan_celeba==1.0.0
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
 

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run stargan_celeba --image "/PATH/TO/IMAGE" --style "target_attribute"
    ```
  - **参数**

    - image ：指定图片路径。

    - style 指定拟转换的属性，可选择 "Black_Hair", "Blond_Hair", "Brown_Hair", "Female", "Male", "Aged" 中的一个。


- ### 2、预测代码示例

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

