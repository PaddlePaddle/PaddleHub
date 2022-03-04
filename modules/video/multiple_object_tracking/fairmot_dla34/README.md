# fairmot_dla34

|模型名称|fairmot_dla34|
| :--- | :---: |
|类别|视频 - 多目标追踪|
|网络|CenterNet|
|数据集|Caltech Pedestrian+CityPersons+CUHK-SYSU+PRW+ETHZ+MOT17|
|是否支持Fine-tuning|否|
|模型大小|125MB|
|最新更新日期|2021-08-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
  <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/131989578-ec06e18f-e122-40b0-84d2-8772fd35391a.gif"  hspace='10'/> <br />
  </p>

- ### 模型介绍

  - FairMOT以Anchor Free的CenterNet检测器为基础，克服了Anchor-Based的检测框架中anchor和特征不对齐问题，深浅层特征融合使得检测和ReID任务各自获得所需要的特征，并且使用低维度ReID特征，提出了一种由两个同质分支组成的简单baseline来预测像素级目标得分和ReID特征，实现了两个任务之间的公平性，并获得了更高水平的实时多目标跟踪精度。

  - 更多详情参考：[FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/abs/2004.01888)



## 二、安装

- ### 1、环境依赖  

  - paddledet >= 2.2.0

  - opencv-python

- ### 2、安装

  - ```shell
    $ hub install fairmot_dla34
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
  - 在windows下安装，由于paddledet package会依赖cython-bbox以及pycocotools, 这两个包需要windows用户提前装好，可参考[cython-bbox安装](https://blog.csdn.net/qq_24739717/article/details/105588729)和[pycocotools安装](https://github.com/PaddlePaddle/PaddleX/blob/release/1.3/docs/install.md#pycocotools安装问题)
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a video file
    $ hub run fairmot_dla34 --video_stream "/PATH/TO/VIDEO"
    ```
  - 通过命令行方式实现多目标追踪模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    tracker = hub.Module(name="fairmot_dla34")
    # Read from a video file
    tracker.tracking('/PATH/TO/VIDEO', output_dir='mot_result', visualization=True,
                        draw_threshold=0.5, use_gpu=False)
    # or read from a image stream
    # with tracker.stream_mode(output_dir='image_stream_output', visualization=True, draw_threshold=0.5, use_gpu=True):
    #    tracker.predict([images])
    ```

- ### 3、API

  - ```python
    def tracking(video_stream,
                 output_dir='',
                 visualization=True,
                 draw_threshold=0.5,
                 use_gpu=False)
    ```
    - 视频预测API，完成对视频内容的多目标追踪，并存储追踪结果。

    - **参数**

      - video_stream (str): 视频文件的路径; <br/>
      - output_dir (str): 结果保存路径的根目录，默认为当前目录； <br/>
      - visualization (bool): 是否保存追踪结果；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - draw\_threshold (float): 预测置信度的阈值。

  - ```python
    def stream_mode(output_dir='',
                    visualization=True,
                    draw_threshold=0.5,
                    use_gpu=False)
    ```
    - 进入图片流预测模式API，在该模式中完成对图片流的多目标追踪，并存储追踪结果。

    - **参数**

      - output_dir (str): 结果保存路径的根目录，默认为当前目录； <br/>
      - visualization (bool): 是否保存追踪结果；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - draw\_threshold (float): 预测置信度的阈值。

  - ```python
    def predict(images: list = [])
    ```
    - 对图片进行预测的API, 该接口必须在stream_mode API被调用后使用。

    - **参数**

      - images (list): 待预测的图片列表。


## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install fairmot_dla34==1.0.0
    ```
