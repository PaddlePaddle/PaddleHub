# reading_pictures_writing_poems

| 模型名称            | reading_pictures_writing_poems |
| :------------------ | :----------------------------: |
| 类别                |         文本-文本生成          |
| 网络                |           多网络级联           |
| 数据集              |               -                |
| 是否支持Fine-tuning |               否               |
| 模型大小            |             3.16K              |
| 最新更新日期        |           2021-04-26           |
| 数据指标            |               -                |

## 一、模型基本信息

- ### 应用效果展示

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133189274-ff86801f-017f-460e-adb0-1d381d74aff6.jpg" width="300">
</p>
  
  - 输入以上图片生成的古诗是：

     - 蕾蕾海河海，岳峰岳麓蔓。
     - 不萌枝上春，自结心中线。

- ### 模型介绍
  - 看图写诗（reading_pictures_writing_poems），该模型可自动根据图像生成古诗词。该PaddleHub Module支持预测。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 1.8.2

  - paddlehub >= 1.8.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

  - translate

    - ```shell
      $ pip install translate
      ```

- ### 2、安装

  - ```shell
    $ hub install reading_pictures_writing_poems
    ```
    
    - 本模型还需要用到xception71_imagenet, ernie_gen_couplet, ernie_gen_poetry这3个模型
    - 若您未安装这3个模型，代码运行时会自动帮您下载
    
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
   | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测
    
  - ```shell
    $ hub run reading_pictures_writing_poems --input_image "scenery.jpg"
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
 
    readingPicturesWritingPoems = hub.Module(name="reading_pictures_writing_poems")
    results = readingPicturesWritingPoems.WritingPoem(image = "scenery.jpg", use_gpu=False)
 
    for result in results:
        print(result)
    ```
  
- ### 3、API

  - ```python
    def WritingPoem(image, use_gpu=False):
    ```

     - 看图写诗预测接口，预测输入一张图像，输出一首古诗词
     - **参数**
         - image(str): 待检测的图片路径
         - use_gpu (bool): 是否使用 GPU
     - **返回**
         - results (list[dict](https://www.paddlepaddle.org.cn/hubdetail?name=reading_pictures_writing_poems&en_category=TextGeneration)): 识别结果的列表，列表中每一个元素为 dict，关键字有 image，Poetrys， 其中： image字段为原输入图片的路径，Poetrys字段为输出的古诗词

## 四、服务部署

- 本模型不支持hub serving


## 五、更新历史

* 1.0.0

  初始发布
