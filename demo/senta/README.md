# Senta 情感分析

本示例展示如何使用PaddleHub Senta Module进行预测。

Senta是百度NLP开放的中文情感分析模型，可以用于进行中文句子的情感分析，输出结果为`{正向/中性/负向}`中的一个，关于模型更多信息参见[Senta](https://www.paddlepaddle.org.cn/hubdetail?name=senta_bilstm&en_category=SentimentAnalysis), 本示例代码选择的是Senta-BiLSTM模型。

## 命令行方式预测

```shell
$ hub run senta_bilstm --input_text "这家餐厅很好吃"
$ hub run senta_bilstm --input_file test.txt
```

test.txt 存放待预测文本， 如：

```text
这家餐厅很好吃
这部电影真的很差劲
```

## 通过python API预测

`senta_demo.py`给出了使用python API调用Module预测的示例代码
通过以下命令试验下效果

```shell
python senta_demo.py
```

## 通过PaddleHub Finetune API微调
`senta_finetune.py` 给出了如何使用Senta模型的句子特征进行Fine-tuning的实例代码。
可以运行以下命令在ChnSentiCorp数据集上进行Fine-tuning。

```shell
$ sh run_finetune.sh
```

同时，我们在AI Studio上提供了IPython NoteBook形式的demo，您可以直接在平台上在线体验，链接如下：

|预训练模型|任务类型|数据集|AIStudio链接|备注|
|-|-|-|-|-|
|ERNIE|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216764)||
|ERNIE|文本分类|中文新闻分类数据集THUNEWS|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216649)|本教程讲述了如何将自定义数据集加载，并利用Finetune API完成文本分类迁移学习。|
|ERNIE|序列标注|中文序列标注数据集MSRA_NER|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216787)||
|ERNIE|序列标注|中文快递单数据集Express|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216683)|本教程讲述了如何将自定义数据集加载，并利用Finetune API完成序列标注迁移学习。|
|ERNIE Tiny|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215599)||
