# videotag_tsn_lstm

|模型名称|videotag_tsn_lstm|
| :--- | :---: | 
|类别|视频-视频分类|
|网络|TSN + AttentionLSTM|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|409MB|
|最新更新日期|2021-02-26|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - videotag_tsn_lstm是一个基于千万短视频预训练的视频分类模型，可直接预测短视频的中文标签。模型分为视频特征抽取和序列建模两个阶段，前者使用TSN网络提取视频特征，后者基于前者输出使用AttentionLSTM网络进行序列建模实现分类。模型基于百度实际短视频场景中的大规模数据训练得到，在实际业务中取得89.9%的Top-1精度，同时具有良好的泛化能力，适用于多种短视频中文标签分类场景。该PaddleHub Module可支持预测。


<p align="center">
<img src="https://paddlehub.bj.bcebos.com/model/video/video_classifcation/VideoTag_TSN_AttentionLSTM.png"  width = "450"  hspace='10'/> <br />
</p>

  - 具体网络结构可参考论文：[TSN](https://arxiv.org/abs/1608.00859) 和 [AttentionLSTM](https://arxiv.org/abs/1503.08909)。



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.7.2
  
  - paddlehub >= 1.6.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install videotag_tsn_lstm
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)




## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    hub run videotag_tsn_lstm --input_path 1.mp4 --use_gpu False
    ```
    
  - 示例文件下载：
    - [1.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/1.mp4)
    - [2.mp4](https://paddlehub.bj.bcebos.com/model/video/video_classifcation/2.mp4)

  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    videotag = hub.Module(name="videotag_tsn_lstm")

    # execute predict and print the result
    results = videotag.classify(paths=["1.mp4","2.mp4"], use_gpu=False)  # 示例文件请在上方下载
    print(results)
    
    #[{'path': '1.mp4', 'prediction': {'训练': 0.9771281480789185, '蹲': 0.9389840960502625, '杠铃': 0.8554490804672241, '健身房': 0.8479971885681152}}, {'path': '2.mp4', 'prediction': {'舞蹈': 0.8504238724708557}}]


    ```
    
- ### 3、API

  - ```python
    def classify(paths,
                 use_gpu=False,
                 threshold=0.5,
                 top_k=10)
    ```    

    - 用于视频分类预测

    - **参数**

      - paths(list\[str\])：mp4文件路径
      - use_gpu(bool)：是否使用GPU预测，默认为False
      - threshold(float)：预测结果阈值，只有预测概率大于阈值的类别会被返回，默认为0.5
      - top_k(int): 返回预测结果的前k个，默认为10

    - **返回**

      - results(list\[dict\]): result中的每个元素为对应输入的预测结果，预测单个mp4文件时仅有1个元素。每个预测结果为dict，包含mp4文件路径path及其分类概率。


## 五、更新历史

* 1.0.0

  初始发布
  
  - ```shell
    $ hub install videotag_tsn_lstm==1.0.0
    ```
