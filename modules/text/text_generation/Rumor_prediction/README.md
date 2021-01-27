## 概述


Rumor_prediction是预测语句是否为谣言的模型。

## 命令行预测

```shell
$ hub run Rumor_prediction --input_text='兴仁县今天抢小孩没抢走，把孩子母亲捅了一刀，看见这车的注意了，真事，车牌号辽HFM055！！！！！赶紧散播！ 都别带孩子出去瞎转悠了 尤其别让老人自己带孩子出去 太危险了 注意了！！！！辽HFM055北京现代朗动，在各学校门口抢小孩！！！110已经 证实！！全市通缉！！'
```

## API

```python
def Rumor(texts, use_gpu=False):
```

预测API，预测语句是否为谣言。

**参数**

* texts (list\[str\]): 想要预测是否为谣言的语句；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA\_VISIBLE\_DEVICES环境变量**；

**返回**

* results (list[dict]): 预测结果的列表，列表中每一个元素为 dict，各字段为：

    - content(str):输入文本内容
    - prediction(str):预测结果
    - probability(float):预测结果概率

**代码示例**

```python
import paddlehub as hub

module = hub.Module(name="Rumor_prediction")

test_texts = ['兴仁县今天抢小孩没抢走，把孩子母亲捅了一刀，看见这车的注意了，真事，车牌号辽HFM055！！！！！赶紧散播！ 都别带孩子出去瞎转悠了 尤其别让老人自己带孩子出去 太危险了 注意了！！！！辽HFM055北京现代朗动，在各学校门口抢小孩！！！110已经 证实！！全市通缉！！']
results = module.Rumor(texts=test_texts, use_gpu=True)
print(results)
```


### 依赖

paddlepaddle >= 2.0.0rc1

paddlehub >= 2.0.0rc0
