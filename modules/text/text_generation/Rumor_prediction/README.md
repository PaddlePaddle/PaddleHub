## 概述


ernie_gen_leave是基于ERNIE-GEN进行微调的模型，该模型的主要功能为生成请假条。输出一个关键词，给出你的请假理由。

## 命令行预测

```shell
$ hub run ernie_gen_leave --input_text="理由" --use_gpu True --beam_width 5
```

## API

```python
def generate(texts, use_gpu=False, beam_width=5):
```

预测API，输入关键字给出请假理由。

**参数**

* texts (list\[str\]): 请假关键字；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA\_VISIBLE\_DEVICES环境变量**；
* beam\_width: beam search宽度，决定输出多少理由的数量。

**返回**

* results (list\[list\]\[str\]): 输出请假理由。

**代码示例**

```python
import paddlehub as hub

module = hub.Module(name="ernie_gen_leave")

test_texts = ["理由"]
results = module.generate(texts=test_texts, use_gpu=False, beam_width=2)
for result in results:
    print(result)
```


## 查看代码

https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.0.0-rc/modules/text/text_generation/ernie_gen_leave

### 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.7.0
