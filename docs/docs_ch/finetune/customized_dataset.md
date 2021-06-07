# 自定义数据

训练一个新任务时，如果从零开始训练时，这将是一个耗时的过程，并且效果可能达不到理想的效果，此时您可以利用PaddleHub提供的预训练模型进行具体任务的Fine-tune。您只需要对自定义数据进行相应的预处理，随后输入预训练模型中，即可得到相应的结果。请参考如下内容设置数据集的结构。

## 一、图像分类数据集

利用PaddleHub迁移分类任务使用自定义数据时，需要切分数据集，将数据集切分为训练集、验证集和测试集。

### 数据准备

需要三个文本文件来记录对应的图片路径和标签，此外还需要一个标签文件用于记录标签的名称。
```
├─data: 数据目录
  ├─train_list.txt：训练集数据列表
  ├─test_list.txt：测试集数据列表
  ├─validate_list.txt：验证集数据列表
  ├─label_list.txt：标签列表
  └─...
```
训练/验证/测试集的数据列表文件的格式如下
```
图片1路径 图片1标签
图片2路径 图片2标签
...
```
label_list.txt的格式如下
```
分类1名称
分类2名称
...
```

示例：
以[Flower数据集](../reference/datasets.md)为示例，train_list.txt/test_list.txt/validate_list.txt内容如下示例
```
roses/8050213579_48e1e7109f.jpg 0
sunflowers/45045003_30bbd0a142_m.jpg 3
daisy/3415180846_d7b5cced14_m.jpg 2
```

label_list.txt内容如下：
```
roses
tulips
daisy
sunflowers
dandelion
```

### 数据集加载

数据集的准备代码可以参考 [flowers.py](../../paddlehub/datasets/flowers.py)。`hub.datasets.Flowers()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。具体使用如下：

```python
from paddlehub.datasets import Flowers

flowers = Flowers(transforms)
flowers_validate = Flowers(transforms, mode='val')
```
* `transforms`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, `val`， 默认为`train`。

## 二、图像着色数据集

利用PaddleHub迁移着色任务使用自定义数据时，需要切分数据集，将数据集切分为训练集和测试集。

### 数据准备

需要将准备用于着色训练和测试的彩色图像分成训练集数据和测试集数据。
```
├─data: 数据目录
  ├─train：训练集数据
      |-图片文件夹1
      |-图片文件夹2
      |-……
      |-图片1
      |-图片2
      |-……

  ├─test：测试集数据
    |-图片文件夹1
    |-图片文件夹2
    |-……
    |-图片1
    |-图片2
    |-……
  └─……
```

示例：
PaddleHub为用户提供了用于着色的数据集`Canvas数据集`， 它由1193张莫奈风格和400张梵高风格的图像组成，以[Canvas数据集](../reference/datasets.md)为示例，train文件夹内容如下:

```
├─train：训练集数据
      |-monet
          |-图片1
          |-图片2
          |-……  
      |-vango
          |-图片1
          |-图片2
          |-……
```

### 数据集加载

数据集的准备代码可以参考 [canvas.py](../../paddlehub/datasets/canvas.py)。`hub.datasets.Canvas()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。具体使用如下：

```python
from paddlehub.datasets import Canvas

color_set = Canvas(transforms, mode='train')
```
* `transforms`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, 默认为`train`。

## 三、风格迁移数据集

利用PaddleHub进行风格迁移任务使用自定义数据时，需要切分数据集，将数据集切分为训练集和测试集。

### 数据准备

需要将准备用于风格迁移的彩色图像分成训练集和测试集数据。

```
├─data: 数据目录
  ├─train：训练集数据
      |-图片文件夹1
      |-图片文件夹2
      |-...
      |-图片1
      |-图片2
      |-...

  ├─test：测试集数据
    |-图片文件夹1
    |-图片文件夹2
    |-...
    |-图片1
    |-图片2
    |-...
  |- 21styles
    ｜-图片1
    ｜-图片2
  └─...
```

示例：
PaddleHub为用户提供了用于风格迁移的数据集`MiniCOCO数据集`, 训练集数据和测试集数据来源于COCO2014， 其中训练集有2001张图片，测试集有200张图片。 `21styles`文件夹下存放着21张不同风格的图片，用户可以根据自己的需求更换不同风格的图片。以[MiniCOCO数据集](../reference/datasets.md)为示例，train文件夹内容如下:

```
├─train：训练集数据
      |-train
          |-图片1
          |-图片2
          |-……  
      |-test
          |-图片1
          |-图片2
          |-……
      |-21styles
          |-图片1
          |-图片2
          |-……
```

### 数据集加载

数据集的准备代码可以参考 [minicoco.py](../../paddlehub/datasets/minicoco.py)。`hub.datasets.MiniCOCO()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。具体使用如下：

```python
from paddlehub.datasets import MiniCOCO

ccolor_set = MiniCOCO(transforms, mode='train')
```
* `transforms`: 数据预处理方式。
* `mode`: 选择数据模式，可选项有 `train`, `test`, 默认为`train`。

## 四、文本分类数据集

利用PaddleHub进行文本分类任务使用自定义数据时，需要切分数据集，将数据集切分为训练集和测试集。

### 数据准备

#### 1. 设置数据集目录

用户需要将数据集目录设定为如下格式：
```shell
├──data: 数据目录
   ├── train.txt: 训练集数据
   ├── dev.txt: 验证集数据
   └── test.txt: 测试集数据
```

#### 2. 设置文件格式和内容

训练/验证/测试集的数据文件的编码格式建议为utf8格式。内容的第一列是文本类别标签，第二列为文本内容，列与列之间以Tab键分隔。建议在数据集文件第一行填写列说明"label"和"text_a"，中间以Tab键分隔，示例如下：
```shell
label    text_a
房产    昌平京基鹭府10月29日推别墅1200万套起享97折
教育    贵州2011高考录取分数线发布理科一本448分
社会    众多白领因集体户口面临结婚难题
...
```

### 数据集加载

加载文本分类的自定义数据集，用户仅需要继承基类TextClassificationDataset，修改数据集存放地址以及类别即可，具体可以参考如下代码：

```python
from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset

class MyDataset(TextClassificationDataset):
    # 数据集存放目录
    base_path = '/path/to/dataset'
    # 数据集的标签列表
    label_list=['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚']

    def __init__(self, tokenizer, max_seq_len: int = 128, mode: str = 'train'):
        if mode == 'train':
            data_file = 'train.txt'
        elif mode == 'test':
            data_file = 'test.txt'
        else:
            data_file = 'dev.txt'
        super().__init__(
            base_path=self.base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=self.label_list,
            is_file_with_header=True)


# 选择所需要的模型，获取对应的tokenizer
import paddlehub as hub
model = hub.Module(name='ernie_tiny', task='seq-cls', num_classes=len(MyDataset.label_list))
tokenizer = model.get_tokenizer()

# 实例化训练集
train_dataset = MyDataset(tokenizer)
```

至此用户可以通过MyDataset实例化获取对应的数据集，可以通过hub.Trainer对预训练模型`model`完成文本分类任务，详情可参考[PaddleHub文本分类demo](https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.0.0-beta/demo/text_classification)。

## 五、序列标注数据集

利用PaddleHub进行序列标注任务使用自定义数据时，需要切分数据集，将数据集切分为训练集和测试集。

### 数据准备

#### 1. 设置数据集目录

用户需要将数据集目录设定为如下格式：
```shell
├──data: 数据目录
   ├── train.txt: 训练集数据
   ├── dev.txt: 验证集数据
   └── test.txt: 测试集数据
```

#### 2. 设置文件格式和内容

训练/验证/测试集的数据文件的编码格式建议为utf8格式。内容的第一列是文本内容, 第二列为文本中每个token对应的标签。需要注意的是，在文本和标签中，都需使用分隔符(该例子中使用的是斜杠`/`)隔开不同的token。  
列与列之间以Tab键分隔。建议在数据集文件第一行填写列说明"label"和"text_a"，中间以Tab键分隔，示例如下：
```shell
text_a    label
5/月/1/2/日/，/北/京/市/怀/柔/县/民/政/局/、/畜/牧/局/领/导/来/到/驻/守/在/偏/远/山/区/的/武/警/北/京/一/总/队/十/支/队/十/四/中/队/。    O/O/O/O/O/O/B-LOC/I-LOC/I-LOC/B-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/O/B-ORG/I-ORG/I-ORG/O/O/O/O/O/O/O/O/O/O/O/O/B-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/O
他/每/年/还/为/河/北/农/业/大/学/扶/助/多/名/贫/困/学/生/。    O/O/O/O/O/B-ORG/I-ORG/I-ORG/I-ORG/I-ORG/I-ORG/O/O/O/O/O/O/O/O/O
...
```

### 数据准备

加载文本分类的自定义数据集，用户仅需要继承基类SeqLabelingDataset，修改数据集存放地址、类别信息和分隔符即可，具体可以参考如下代码：

```python
from paddlehub.datasets.base_nlp_dataset import SeqLabelingDataset

class MyDataset(SeqLabelingDataset):
    # 数据集存放目录
    base_path = '/path/to/dataset'
    # 数据集的标签列表
    label_list = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]
    label_map = {idx: label for idx, label in enumerate(label_list)}
    # 数据文件使用的分隔符
    split_char = '/'

    def __init__(self, tokenizer, max_seq_len: int = 128, mode: str = 'train'):
        if mode == 'train':
            data_file = 'train.txt'
        elif mode == 'test':
            data_file = 'test.txt'
        else:
            data_file = 'dev.txt'
        super().__init__(
                    base_path=self.base_path,
                    tokenizer=tokenizer,
                    max_seq_len=max_seq_len,
                    mode=mode,
                    data_file=data_file,
                    label_file=None,
                    label_list=self.label_list,
                    split_char=self.split_char,
                    is_file_with_header=True)

# 选择所需要的模型，获取对应的tokenizer
import paddlehub as hub
model = hub.Module(name='ernie_tiny', task='token-cls', label_map=MyDataset.label_map)
tokenizer = model.get_tokenizer()

# 实例化训练集
train_dataset = MyDataset(tokenizer)
```

至此用户可以通过MyDataset实例化获取对应的数据集，可以通过hub.Trainer对预训练模型`model`完成系列标注任务，详情可参考[PaddleHub序列标注demo](https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.0.0-beta/demo/sequence_labeling)。
