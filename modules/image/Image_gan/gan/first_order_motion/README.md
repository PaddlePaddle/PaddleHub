# first_order_motion

|模型名称|first_order_motion|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|S3FD|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|343MB|
|最新更新日期|2021-12-24|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/147347145-1a7e84b6-2853-4490-8eaf-caf9cfdca79b.png"  width = "40%"  hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/147347151-d6c5690b-00cd-433f-b82b-3f8bb90bc7bd.gif"  width = "40%"  hspace='10'/>
    <br />
    输入视频
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/147348127-52eb3f26-9b2c-49d5-a4a2-20a31f159802.gif"  width = "40%"  hspace='10'/>
    <br />
    输出视频
     <br />
    </p>

- ### 模型介绍

  - First Order Motion的任务是图像动画/Image Animation，即输入为一张源图片和一个驱动视频，源图片中的人物则会做出驱动视频中的动作。


## 二、安装

- ### 1、环境依赖  
  - paddlepaddle >= 2.1.0
  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install first_order_motion
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run first_order_motion --source_image "/PATH/TO/IMAGE" --driving_video "/PATH/TO/VIDEO"  --use_gpu
    ```
  - 通过命令行方式实现视频驱动生成模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="first_order_motion")
    module.generate(source_image="/PATH/TO/IMAGE", driving_video="/PATH/TO/VIDEO", ratio=0.4, image_size=256, output_dir='./motion_driving_result/', filename='result.mp4', use_gpu=False)
    ```

- ### 3、API

  - ```python
    generate(self, source_image=None, driving_video=None, ratio=0.4, image_size=256, output_dir='./motion_driving_result/', filename='result.mp4', use_gpu=False)
    ```
    - 视频驱动生成API。

    - **参数**
      - source_image (str):  原始图片，支持单人图片和多人图片，视频中人物的表情动作将迁移到该原始图片中的人物上。
      - driving_video (str): 驱动视频，视频中人物的表情动作作为待迁移的对象。
      - ratio (float): 贴回驱动生成的人脸区域占原图的比例, 用户需要根据生成的效果调整该参数，尤其对于多人脸距离比较近的情况下需要调整改参数, 默认为0.4，调整范围是[0.4, 0.5]。
      - image_size (int): 图片人脸大小，默认为256，可设置为512。
      - output\_dir (str): 结果保存的文件夹名； <br/>
      - filename (str): 结果保存的文件名。
      - use\_gpu (bool): 是否使用 GPU；<br/>


## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install first_order_motion==1.0.0
    ```
