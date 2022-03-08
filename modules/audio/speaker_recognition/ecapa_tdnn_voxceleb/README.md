# ecapa_tdnn_voxceleb

|模型名称|ecapa_tdnn_voxceleb|
| :--- | :---: |
|类别|语音-声纹识别|
|网络|ECAPA-TDNN|
|数据集|VoxCeleb|
|是否支持Fine-tuning|否|
|模型大小|79MB|
|最新更新日期|2021-12-30|
|数据指标|EER 0.69%|

## 一、模型基本信息

### 模型介绍

ecapa_tdnn_voxceleb采用了[ECAPA-TDNN](https://arxiv.org/abs/2005.07143)的模型结构，并在[VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)数据集上进行了预训练，在VoxCeleb1的声纹识别测试集([veri_test.txt](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt))上的测试结果为 EER 0.69%，达到了该数据集的SOTA。

<p align="center">
<img src="https://d3i71xaburhd42.cloudfront.net/9609f4817a7e769f5e3e07084db35e46696e82cd/3-Figure2-1.png" hspace='10' height="550"/> <br />
</p>



更多详情请参考
- [VoxCeleb: a large-scale speaker identification dataset](https://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf)
- [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/pdf/2005.07143.pdf)
- [The SpeechBrain Toolkit](https://github.com/speechbrain/speechbrain)


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.2.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ecapa_tdnn_voxceleb
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、预测代码示例

    ```python
    import paddlehub as hub

    model = hub.Module(
        name='ecapa_tdnn_voxceleb',
        threshold=0.25,
        version='1.0.0')

    # 通过下列链接可下载示例音频
    # https://paddlehub.bj.bcebos.com/paddlehub_dev/sv1.wav
    # https://paddlehub.bj.bcebos.com/paddlehub_dev/sv2.wav

    # Speaker Embedding
    embedding = model.speaker_embedding('sv1.wav')
    print(embedding.shape)
    # (192,)

    # Speaker Verification
    score, pred = model.speaker_verify('sv1.wav', 'sv2.wav')
    print(score, pred)
    # [0.16354457], [False]
    ```

- ### 2、API
  - ```python
    def __init__(
        threshold: float,
    )
    ```
    - 初始化声纹模型，确定判别阈值。

    - **参数**

      - `threshold`：设定模型判别声纹相似度的得分阈值，默认为 0.25。

  - ```python
    def speaker_embedding(
        wav: os.PathLike,
    )
    ```
    - 获取输入音频的声纹特征

    - **参数**

      - `wav`：输入的说话人的音频文件，格式为`*.wav`。

    - **返回**

      - 输出纬度为 (192,) 的声纹特征向量。

  - ```python
    def speaker_verify(
        wav1: os.PathLike,
        wav2: os.PathLike,
    )
    ```
    - 对比两段音频，分别计算其声纹特征的相似度得分，并判断是否为同一说话人。

    - **参数**

      - `wav1`：输入的说话人1的音频文件，格式为`*.wav`。
      - `wav2`：输入的说话人2的音频文件，格式为`*.wav`。

    - **返回**

      - 返回声纹相似度得分[-1, 1]和预测结果。


## 四、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install ecapa_tdnn_voxceleb
  ```
