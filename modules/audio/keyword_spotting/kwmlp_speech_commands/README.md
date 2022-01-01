# kwmlp_speech_commands

|模型名称|kwmlp_speech_commands|
| :--- | :---: |
|类别|语音-语言识别|
|网络|Keyword-MLP|
|数据集|Google Speech Commands V2|
|是否支持Fine-tuning|否|
|模型大小|1.6MB|
|最新更新日期|2022-01-04|
|数据指标|ACC 97.56%|

## 一、模型基本信息

### 模型介绍

kwmlp_speech_commands采用了 [Keyword-MLP](https://arxiv.org/pdf/2110.07749v1.pdf) 的轻量级模型结构，并在 [Google Speech Commands V2](https://arxiv.org/abs/1804.03209) 数据集上进行了预训练，在其测试集的测试结果为 ACC 97.56%。

<p align="center">
<img src="https://d3i71xaburhd42.cloudfront.net/fa690a97f76ba119ca08fb02fa524a546c47f031/2-Figure1-1.png" hspace='10' height="550"/> <br />
</p>


更多详情请参考
- [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209)
- [ATTENTION-FREE KEYWORD SPOTTING](https://arxiv.org/pdf/2110.07749v1.pdf)
- [Keyword-MLP](https://github.com/AI-Research-BD/Keyword-MLP)


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.2.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install kwmlp_speech_commands
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、预测代码示例

    ```python
    import paddlehub as hub

    model = hub.Module(
        name='kwmlp_speech_commands',
        version='1.0.0')

    # 通过下列链接可下载示例音频
    # https://paddlehub.bj.bcebos.com/paddlehub_dev/go.wav

    # Keyword spotting
    score, label = model.keyword_recognize('no.wav')
    print(score, label)
    # [0.89498246] no
    score, label = model.keyword_recognize('go.wav')
    print(score, label)
    # [0.8997176] go
    score, label = model.keyword_recognize('one.wav')
    print(score, label)
    # [0.88598305] one
    ```

- ### 2、API
  - ```python
    def keyword_recognize(
        wav: os.PathLike,
    )
    ```
    - 检测音频中包含的关键词。

    - **参数**

      - `wav`：输入的包含关键词的音频文件，格式为`*.wav`。

    - **返回**

      - 输出结果的得分和对应的关键词标签。


## 四、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install kwmlp_speech_commands
  ```
