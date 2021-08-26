# jde_darknet53_30e_1088x608

|模型名称|jde_darknet53_30e_1088x608|
| :--- | :---: | 
|类别|视频 - 多目标追踪|
|网络|YOLOv3|
|数据集|Caltech Pedestrian+CityPersons+CUHK-SYSU+PRW+ETHZ+MOT17|
|是否支持Fine-tuning|否|
|模型大小|420MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
  <p align="center">
  <img src="demo/mot16_jde.gif"  width = "450" height = "300" hspace='10'/> <br />
  </p>

- ### 模型介绍

  - JDE(Joint Detection and Embedding)是在一个单一的共享神经网络中同时学习目标检测任务和embedding任务，并同时输出检测结果和对应的外观embedding匹配的算法。JDE原论文是基于Anchor Base的YOLOv3检测器新增加一个ReID分支学习embedding，训练过程被构建为一个多任务联合学习问题，兼顾精度和速度。

  - 更多详情参考：[Towards Real-Time Multi-Object Tracking](https://arxiv.org/abs/1909.12605)



## 二、安装

- ### 1、环境依赖     

  - paddlepaddle >= 2.1.0    

  - paddlehub >= 1.6.0     

  - ppdet >= 2.1.0

  - ffmpeg                       

- ### 2、安装

  - ```shell
    $ hub install jde_darknet53_30e_1088x608
    ```
  
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a video file
    $ hub run jde_darknet53_30e_1088x608 --video_stream "/PATH/TO/VIDEO"

    # Read from a camera device
    $ hub run jde_darknet53_30e_1088x608 --video_stream 0 --from_device
    ```


- ### 2、代码示例

  - ```python
    import paddlehub as hub

    jde_tracker = hub.Module(name="jde_darknet53_30e_1088x608")
    # Read from a video file
    jde_tracker.mot_predict('/PATH/TO/VIDEO', output_dir='mot_result', visualization=True, \
                            draw_threshold=0.5, use_gpu=False, from_device=False)
    # or read from a camera device
    # jde_tracker.mot_predict('0', output_dir='mot_result', visualization=True, \
                            draw_threshold=0.5, use_gpu=False, from_device=True)
    ```

- ### 3、API

  - ```python
    def mot_predict(video_stream,
                    output_dir='',
                    visualization=True,
                    draw_threshold=0.5,
                    use_gpu=False,
                    from_device=False
                    )
    ```
    - 预测API，完成对视频内容的多目标追踪，并存储追踪结果。

    - **参数**

      - video_stream (str): 视频流的来源，可以是视频文件的路径，也可以是本机摄像设备的设备索引号； <br/>
      - output_dir (str): 结果保存路径的根目录，默认为当前目录； <br/>
      - visualization (bool): 是否保存追踪结果；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - draw\_threshold (float): 预测置信度的阈值；<br/>      
      - from_device (bool): 是否从摄像设备读取视频，如若是，video_stream的输入会被当成设备索引号。

## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install jde_darknet53_30e_1088x608==1.0.0
    ```