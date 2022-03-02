# panns_cnn6

|模型名称|panns_cnn6|
| :--- | :---: |
|类别|语音-声音分类|
|网络|PANNs|
|数据集|Google Audioset|
|是否支持Fine-tuning|是|
|模型大小|29MB|
|最新更新日期|2021-06-15|
|数据指标|mAP 0.343|

## 一、模型基本信息

### 模型介绍

`panns_cnn6`是一个基于[Google Audioset](https://research.google.com/audioset/)数据集训练的声音分类/识别的模型。该模型主要包含4个卷积层和2个全连接层，模型参数为4.5M。经过预训练后，可以用于提取音频的embbedding，维度是512。

更多详情请参考：[PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/pdf/1912.10211.pdf)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install panns_cnn6
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、预测代码示例

  - ```python
    # ESC50声音分类预测
    import librosa
    import paddlehub as hub
    from paddlehub.datasets import ESC50

    sr = 44100 # 音频文件的采样率
    wav_file = '/PATH/TO/AUDIO' # 用于预测的音频文件路径
    checkpoint = 'model.pdparams' # 用于预测的模型参数

    label_map = {idx: label for idx, label in enumerate(ESC50.label_list)}

    model = hub.Module(
        name='panns_cnn6',
        version='1.0.0',
        task='sound-cls',
        num_class=ESC50.num_class,
        label_map=label_map,
        load_checkpoint=checkpoint)

    data = [librosa.load(wav_file, sr=sr)[0]]
    result = model.predict(
        data,
        sample_rate=sr,
        batch_size=1,
        feat_type='mel',
        use_gpu=True)

    print('File: {}\tLable: {}'.format(wav_file, result[0]))
    ```

  - ```python
    # Audioset Tagging
    import librosa
    import numpy as np
    import paddlehub as hub

    def show_topk(k, label_map, file, result):
        """
        展示topk的分的类别和分数。
        """
        result = np.asarray(result)
        topk_idx = (-result).argsort()[:k]
        msg = f'[{file}]\n'
        for idx in topk_idx:
            label, score = label_map[idx], result[idx]
            msg += f'{label}: {score}\n'
        print(msg)

    sr = 44100 # 音频文件的采样率
    wav_file = '/PATH/TO/AUDIO' # 用于预测的音频文件路径
    label_file = './audioset_labels.txt' # audioset标签文本文件
    topk = 10 # 展示的topk数

    label_map = {}
    with open(label_file, 'r') as f:
        for i, l in enumerate(f.readlines()):
            label_map[i] = l.strip()

    model = hub.Module(
        name='panns_cnn6',
        version='1.0.0',
        task=None)

    data = [librosa.load(wav_file, sr=sr)[0]]
    result = model.predict(
        data,
        sample_rate=sr,
        batch_size=1,
        feat_type='mel',
        use_gpu=True)

    show_topk(topk, label_map, wav_file, result[0])
    ```

- ### 2、API

  - ```python
    def __init__(
            task,
            num_class=None,
            label_map=None,
            load_checkpoint=None,
            **kwargs,
    )
    ```
    - 创建Module对象。

    - **参数**
      - `task`： 任务名称，可为`sound-cls`或者`None`。`sound-cls`代表声音分类任务，可以对声音分类的数据集进行finetune；为`None`时可以获取预训练模型对音频进行分类/Tagging。
      - `num_classes`：声音分类任务的类别数，finetune时需要指定，数值与具体使用的数据集类别数一致。
      - `label_map`：预测时的类别映射表。
      - `load_checkpoint`：使用PaddleHub Fine-tune api训练保存的模型参数文件路径。
      - `**kwargs`：用户额外指定的关键字字典类型的参数。

  - ```python
    def predict(
            data,
            sample_rate,
            batch_size=1,
            feat_type='mel',
            use_gpu=False
    )
    ```
    - 模型预测，输入为音频波形数据，输出为分类标签。

    - **参数**
      - `data`： 待预测数据，格式为\[waveform1, wavwform2…,\]，其中每个元素都是一个一维numpy列表，是音频的波形采样数值列表。
      - `sample_rate`：音频文件的采样率。
      - `feat_type`：音频特征的种类选取，当前支持`'mel'`(详情可查看[Mel-frequency cepstrum](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum))和原始波形特征`'raw'`。
      - `batch_size`：模型批处理大小。
      - `use_gpu`：是否使用gpu，默认为False。对于GPU用户，建议开启use_gpu。

    - **返回**
      - `results`：list类型，不同任务类型的返回结果如下
      - 声音分类(task参数为`sound-cls`)：列表里包含每个音频文件的分类标签。
      - Tagging(task参数为`None`)：列表里包含每个音频文件527个类别([Audioset标签](https://research.google.com/audioset/))的得分。

    详情可参考PaddleHub示例：
    - [AudioClassification](https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.0/demo/audio_classification)

## 四、更新历史

* 1.0.0

  初始发布，动态图版本模型，支持声音分类`sound-cls`任务的fine-tune和基于Audioset Tagging预测。

  ```shell
  $ hub install panns_cnn6
  ```
