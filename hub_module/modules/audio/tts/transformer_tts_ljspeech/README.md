## 概述

TansformerTTS 是使用了 Transformer 结构的端到端语音合成模型，对 Transformer 和 Tacotron2 进行了融合，取得了令人满意的效果。因为删除了 RNN 的循环连接，可并行的提供 decoder 的输入，进行并行训练，大大提升了模型的训练速度。transformer_tts_ljspeech是基于ljspeech英文语音数据集预训练得到的英文TTS模型，仅支持预测。

<p align="center">
<img src="https://github.com/PaddlePaddle/Parakeet/blob/develop/examples/transformer_tts/images/model_architecture.jpg" hspace='10'/> <br />
</p>

更多详情参考论文[Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)

## 命令行预测

```shell
$ hub run transformer_tts_ljspeech --input_text="Life was like a box of chocolates, you never know what you're gonna get." --use_gpu True --vocoder griffin-lim
```

## API

```python
def synthesize(texts, use_gpu=False, vocoder="griffin-lim"):
```

预测API，由输入文本合成对应音频波形。

**参数**

* texts (list\[str\]): 待预测文本；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA\_VISIBLE\_DEVICES环境变量**；
* vocoder: 指定声码器，可选 "griffin-lim"或"waveflow"

**返回**

* wavs (list): 语音合成结果列表，列表中每一个元素为对应输入文本的音频波形，可使用`soundfile.write`进一步处理或保存。
* sample\_rate (int): 合成音频的采样率。

**代码示例**

```python
import paddlehub as hub
import soundfile as sf

# Load transformer_tts_ljspeech module.
module = hub.Module(name="transformer_tts_ljspeech")

# Predict sentiment label
test_texts = ["Life was like a box of chocolates, you never know what you're gonna get."]
wavs, sample_rate = module.synthesize(texts=test_texts, use_gpu=True, vocoder="waveflow")
for index, wav in enumerate(wavs):
    sf.write(f"{index}.wav", wav, sample_rate)
```

## 服务部署

PaddleHub Serving 可以部署在线服务。

### 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m transformer_tts_ljspeech
```

这样就完成了一个服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

### 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

import soundfile as sf

# 发送HTTP请求

data = {'texts':['Simple as this proposition is, it is necessary to be stated',
                 'Parakeet stands for Paddle PARAllel text-to-speech toolkit'],
        'use_gpu':False}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/transformer_tts_ljspeech"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 保存结果
result = r.json()["results"]
wavs = result["wavs"]
sample_rate = result["sample_rate"]
for index, wav in enumerate(wavs):
    sf.write(f"{index}.wav", wav, sample_rate)
```

## 查看代码

https://github.com/PaddlePaddle/Parakeet

### 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.7.0


## 更新历史

* 1.0.0

  初始发布
