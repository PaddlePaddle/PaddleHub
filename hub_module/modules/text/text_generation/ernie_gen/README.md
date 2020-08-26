## 概述

ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入span-by-span 生成任务，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过填充式生成机制和噪声感知机制来缓解曝光偏差问题。此外, ERNIE-GEN 采样多片段-多粒度目标文本采样策略, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。ernie_gen module是一个具备微调功能的module，可以快速完成特定场景module的制作。
<p align="center">
<img src="https://paddlehub.bj.bcebos.com/resources/multi-flow-attention.png" hspace='10'/> <br />
</p>

更多详情参考论文[ERNIE-GEN:An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)。

## API

```python
def finetune(
  train_path,
  dev_path=None,
  save_dir="ernie_gen_result",
  init_ckpt_path=None,
  use_gpu=True,
  max_steps=500,
  batch_size=8,
  max_encode_len=15,
  max_decode_len=15,
  learning_rate=5e-5,
  warmup_proportion=0.1,
  weight_decay=0.1,
  noise_prob=0,
  label_smooth=0,
  beam_width=5,
  length_penalty=1.0,
  log_interval=100,
  save_interval=200,
):
```

微调API，

**参数**

* train_path(str): 训练集路径。训练集的格式应为："序号\t输入文本\t标签"，例如："1\t床前明月光\t疑是地上霜"

* dev_path(str): 验证集路径。验证集的格式应为："序号\t输入文本\t标签"，例如："1\t举头望明月\t低头思故乡"

* save_dir(str): 模型保存以及验证集预测输出路径。

* init_ckpt_path(str): 模型初始化加载路径，可实现增量训练。

* use_gpu(bool): 是否使用GPU。

* max_steps(int): 最大训练步数。

* batch_size(int): 训练时的batch大小。

* max_encode_len(int): 最长编码长度。

* max_decode_len(int): 最长解码长度。

* learning_rate(float): 学习率大小。

* warmup_proportion(float): 学习率warmup比例。

* weight_decay(float): 权值衰减大小。

* noise_prob(float): 噪声概率，详见ernie gen论文。

* label_smooth(float): 标签平滑权重。

* beam_width(int): 验证集预测时的beam大小。

* length_penalty(float): 验证集预测时的长度惩罚权重。

* log_interval(int): 训练时的日志打印间隔步数。

* save_interval(int): 训练时的模型保存间隔部署。验证集将在模型保存完毕后进行预测。

**返回**

* save_path(str): 最后一次保存的模型参数路径。

```python
def export(
  params_path,
  module_name,
  author,
  version="1.0.0",
  summary="",
  author_email="",
  export_path=".")
```

module导出API，通过此API可以一键将训练参数打包为hub module。

**参数**

* params_path(str): 模型参数路径。

* module_name(str): module名称，例如"ernie_gen_couplet"。

* author(str): 作者名称。

* version(str): 版本号。

* summary(str): module的英文简介。

* author_email(str): 作者的邮箱地址。

* export_path(str): module的导出路径。

**代码示例**

```python
import paddlehub as hub

savepath = module.finetune(
    train_path='test_data/train.txt',
    dev_path='test_data/dev.txt',
    max_steps=300,
    batch_size=2
)

module.export(params_path=savepath, module_name="ernie_gen_test", author="test")
```

模型转换完毕之后，可以将导出文件夹移动至 ~/.paddlehub/modules，即可通过以下2种方式调用自制module：

*NOTE:* 下述`$module_name`为export指定的module_name。

1. 命令行预测

```shell
$ hub run $module_name --input_text="输入文本" --use_gpu True --beam_width 5
```

2. API预测
```python
import paddlehub as hub

module = hub.Module(name="$module_name")

test_texts = ["输入文本1", "输入文本2"]
# generate包含3个参数，texts为输入文本列表，use_gpu指定是否使用gpu，beam_width指定beam search宽度。
results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
for result in results:
    print(result)
```

## 查看代码

https://github.com/PaddlePaddle/ERNIE/blob/repro/ernie-gen/

### 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.7.0


## 更新历史

* 1.0.0

  初始发布
