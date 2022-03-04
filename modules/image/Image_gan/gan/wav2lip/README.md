# wav2lip

|模型名称|wav2lip|
| :--- | :---: |
|类别|图像 - 视频生成|
|网络|Wav2Lip|
|数据集|LRS2|
|是否支持Fine-tuning|否|
|模型大小|139MB|
|最新更新日期|2021-12-14|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/146481773-4ec50285-3b13-4a86-84a2-b105787b63d1.png"  width = "40%"  hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/146482210-5f309fc3-7582-452d-bcf5-f2c54b5c8dc8.gif"  width = "40%"  hspace='10'/>
    <br />
    输出视频
     <br />
    </p>


- ### 模型介绍

  - Wav2Lip实现的是视频人物根据输入音频生成与语音同步的人物唇形，使得生成的视频人物口型与输入语音同步。Wav2Lip不仅可以基于静态图像来输出与目标语音匹配的唇形同步视频，还可以直接将动态的视频进行唇形转换，输出与目标语音匹配的视频。Wav2Lip实现唇形与语音精准同步突破的关键在于，它采用了唇形同步判别器，以强制生成器持续产生准确而逼真的唇部运动。此外，它通过在鉴别器中使用多个连续帧而不是单个帧，并使用视觉质量损失（而不仅仅是对比损失）来考虑时间相关性，从而改善了视觉质量。Wav2Lip适用于任何人脸、任何语言，对任意视频都能达到很高都准确率，可以无缝地与原始视频融合，还可以用于转换动画人脸。



## 二、安装

- ### 1、环境依赖  
  - ffmpeg
  - libsndfile
- ### 2、安装

  - ```shell
    $ hub install wav2lip
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run wav2lip --face "/PATH/TO/VIDEO or IMAGE" --audio "/PATH/TO/AUDIO"
    ```
  - 通过命令行方式人物唇形生成模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="wav2lip")
    face_input_path = "/PATH/TO/VIDEO or IMAGE"
    audio_input_path = "/PATH/TO/AUDIO"
    module.wav2lip_transfer(face=face_input_path, audio=audio_input_path, output_dir='./transfer_result/', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    def wav2lip_transfer(face, audio, output_dir ='./output_result/', use_gpu=False, visualization=True):
    ```
    - 人脸唇形生成API。

    - **参数**

      - face (str): 视频或图像文件的路径<br/>
      - audio (str): 音频文件的路径<br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization(bool): 是否保存结果到本地文件夹


## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install wav2lip==1.0.0
    ```
