# Reader

PaddleHub的数据预处理模块Reader对常见的NLP和CV任务进行了抽象。

## NLPReader

### hub.reader.ClassifyReader

#### Class `ClassifyReader`
适用于Transformer预训练模型(ERNIE/BERT)的数据预处理器。

hub.reader.ClassifyReader(
    dataset,
    vocab_path,
    max_seq_len,
    do_lower_case,
    random_seed
)

**参数:**
* `dataset`: hub.dataset中的数据集
* `vocab_path`: 模型词表路径
* `max_seq_len`: 最大序列长度
* `do_lower_case`: 是否讲所有字符中的大写字符转为小写字符
* `random_seed`: 随机种子，默认为None

**返回**

`ClassifyReader`

**示例**
```python
import paddlehub as hub

dataset = hub.dataset.ChnSentiCorp()

reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=args.max_seq_len)

```

------

### hub.reader.LACClassifyReader
#### Class `LACClassifyReader`
以LAC模块为切词器的预处理模块，适用于Senta、ELMo等需要以词粒度分词的任务。

```python
hub.reader.LACClassificationReader(
    dataset,
    vocab_path)
```

**参数:**
* `dataset`: hub.dataset中的数据集
* `vocab_path`: 模型词表路径

**返回**

`LACClassificationReader`

**示例**
```python
import paddlehub as hub

dataset = hub.dataset.ChnSentiCorp()

reader = hub.reader.LACClassifyReader(
        dataset=dataset, vocab_path=module.get_vocab_path())

```


***NOTE***: 使用LACClassificationReader时，安装的lac module版本应该至少为1.0.1


------

### hub.reader.SequenceLabelReader
#### Class `SequenceLabelReader`
适用于Transformer类模型(ERNIE/BERT)的序列标注预处理器。

hub.reader.SequenceLabelReader(
    dataset,
    vocab_path,
    max_seq_len,
    do_lower_case,
    random_seed
)

**参数:**
* `dataset`: hub.dataset中的数据集
* `vocab_path`: 模型词表路径
* `max_seq_len`: 最大序列长度
* `do_lower_case`: 是否讲所有字符中的大写字符转为小写字符
* `random_seed`: 随机种子，默认为None

**返回**

`ClassifyReader`

**示例**
```python
import paddlehub as hub

dataset = hub.dataset.ChnSentiCorp()

reader = hub.reader.SequenceLabelReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    max_seq_len=args.max_seq_len)

```
## CVReader

### hub.reader.cv_reader.ImageClassificationReader

#### Class `ImageClassificationReader`
```python
hub.ImageClassificationReader(
    image_width,
    image_height,
    dataset,
    channel_order="RGB",
    images_mean=None,
    images_std=None,
    data_augmentation=False)
```

适用于图像分类数据的预处理器。会修改输入图像的尺寸、进行标准化处理、图像增广处理等操作。

**参数**
> * image_width: 预期经过reader处理后的图像宽度
> * image_height: 预期经过reader处理后的图像高度
> * dataset: 数据集
> * channel_order: 预期经过reader处理后的图像通道顺序。默认为RGB
> * images_mean: 进行标准化处理时所减均值。默认为None
> * images_std: 进行标准化处理时所除标准差。默认为None
> * data_augmentation: 是否使用图像增广，当开启这个选项时，会对输入数据进行随机变换，包括左右对换，上下倒置，旋转一定的角度，对比度调整等

**返回**

`ImageClassificationReader`

**示例**
```python
import paddlehub as hub

dataset = hub.dataset.Flowers()

data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)
```

#### `data_generator`
为数据集或者指定的数据生成对应的batch reader。

**参数**
> * batch_size: 所返回的batch reader的batch大小
> * phase: 生成什么类型的数据集，phase只能是{train/test/val/dev/predict}中的一个，其中train表示训练集，test表示测试集，val或dev表示验证集，predict表示预测数据。默认为"train"
> * shuffle: 是否打乱数据。默认为False
> * data: 待预测数据，当phase为"predict"时，需要提供将预测数据填入这个字段

**示例**
```python
...
data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)
...
hub.finetune_and_eval(
    task, feed_list=feed_list, data_reader=data_reader, config=config)
```
