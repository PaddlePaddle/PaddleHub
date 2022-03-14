# ecapa_tdnn_common_language

|模型名称|ecapa_tdnn_common_language|
| :--- | :---: |
|类别|语音-语言识别|
|网络|ECAPA-TDNN|
|数据集|CommonLanguage|
|是否支持Fine-tuning|否|
|模型大小|79MB|
|最新更新日期|2021-12-30|
|数据指标|ACC 84.9%|

## 一、模型基本信息

### 模型介绍

ecapa_tdnn_common_language采用了[ECAPA-TDNN](https://arxiv.org/abs/2005.07143)的模型结构，并在[CommonLanguage](https://zenodo.org/record/5036977/)数据集上进行了预训练，在其测试集的测试结果为 ACC 84.9%。

<p align="center">
<img src="https://d3i71xaburhd42.cloudfront.net/9609f4817a7e769f5e3e07084db35e46696e82cd/3-Figure2-1.png" hspace='10' height="550"/> <br />
</p>


更多详情请参考
- [CommonLanguage](https://zenodo.org/record/5036977#.Yc19b5Mzb0o)
- [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/pdf/2005.07143.pdf)
- [The SpeechBrain Toolkit](https://github.com/speechbrain/speechbrain)


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.2.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ecapa_tdnn_common_language
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、预测代码示例

    ```python
    import paddlehub as hub

    model = hub.Module(
        name='ecapa_tdnn_common_language',
        version='1.0.0')

    # 通过下列链接可下载示例音频
    # https://paddlehub.bj.bcebos.com/paddlehub_dev/zh.wav
    # https://paddlehub.bj.bcebos.com/paddlehub_dev/en.wav
    # https://paddlehub.bj.bcebos.com/paddlehub_dev/it.wav

    # Language Identification
    score, label = model.speaker_verify('zh.wav')
    print(score, label)
    # array([0.6214552], dtype=float32), 'Chinese_China'
    score, label = model.speaker_verify('en.wav')
    print(score, label)
    # array([0.37193954], dtype=float32), 'English'
    score, label = model.speaker_verify('it.wav')
    print(score, label)
    # array([0.46913534], dtype=float32), 'Italian'
    ```

- ### 2、API
  - ```python
    def language_identify(
        wav: os.PathLike,
    )
    ```
    - 判断输入人声音频的语言类别。

    - **参数**

      - `wav`：输入的说话人的音频文件，格式为`*.wav`。

    - **返回**

      - 输出结果的得分和对应的语言类别。


## 四、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install ecapa_tdnn_common_language
  ```
