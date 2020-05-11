# 自定义数据

训练一个新任务时，如果从零开始训练时，这将是一个耗时的过程，并且效果可能达不到理想的效果，此时您可以利用PaddleHub提供的预训练模型进行具体任务的Fine-tune。您只需要对自定义数据进行相应的预处理，随后输入预训练模型中，即可得到相应的结果。


## 一、NLP类任务如何自定义数据

本文以预训练模型ERNIE对文本分类任务进行Fine-tune为例，说明如何利用PaddleHub适配自定义数据完成Fine-tune。

### 数据准备

```
├─data: 数据目录
  ├─train.tsv 训练集
  ├─dev.tsv 验证集
  ├─test.tsv 测试集
  └─……
```


相应的数据格式为第一列是文本内容text_a，第二列为文本类别label。列与列之间以Tab键分隔。数据集文件第一行为`text_a    label`（中间以Tab键分隔）。

如果您有两个输入文本text_a、text_b，则第一列为第一个输入文本text_a, 第二列应为第二个输入文本text_b，第三列文本类别label。列与列之间以Tab键分隔。数据集第一行为`text_a    text_b    label`（中间以Tab键分隔）。


```text
text_a    label
15.4寸笔记本的键盘确实爽，基本跟台式机差不多了，蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错    1
房间太小。其他的都一般。。。。。。。。。    0
1.接电源没有几分钟,电源适配器热的不行. 2.摄像头用不起来. 3.机盖的钢琴漆，手不能摸，一摸一个印. 4.硬盘分区不好办.    0
```

### 自定义数据加载
加载文本类自定义数据集，用户仅需要继承基类BaseNLPDatast，修改数据集存放地址以及类别即可。具体使用如下：

**NOTE:**
* 数据集文件编码格式建议为utf8格式。
* 如果相应的数据集文件没有上述的列说明，如train.tsv文件没有第一行的`text_a    label`，则train_file_with_header=False。
* 如果您还有预测数据（没有文本类别），可以将预测数据存放在predict.tsv文件，文件格式和train.tsv类似。去掉label一列即可。
* 分类任务中，数据集的label必须从0开始计数


```python
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

class DemoDataset(BaseNLPDataset):
    """DemoDataset"""
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "path/to/dataset"
        super(DemoDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.tsv",
            dev_file="dev.tsv",
            test_file="test.tsv",
            # 如果还有预测数据（不需要文本类别label），可以放在predict.tsv
            predict_file="predict.tsv",
            train_file_with_header=True,
            dev_file_with_header=True,
            test_file_with_header=True,
            predict_file_with_header=True,
            # 数据集类别集合
            label_list=["0", "1"])
dataset = DemoDataset()
```

之后，您就可以通过DemoDataset()获取自定义数据集了。进而配合ClassifyReader以及预训练模型如ERNIE完成文本分类任务。

## 二、CV类任务如何自定义数据

利用PaddleHub迁移CV类任务使用自定义数据时，用户需要自己切分数据集，将数据集且分为训练集、验证集和测试集。

### 数据准备

需要三个文本文件来记录对应的图片路径和标签，此外还需要一个标签文件用于记录标签的名称。
```
├─data: 数据目录
  ├─train_list.txt：训练集数据列表
  ├─test_list.txt：测试集数据列表
  ├─validate_list.txt：验证集数据列表
  ├─label_list.txt：标签列表
  └─……
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
以[DogCat数据集](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-API:-Dataset#class-hubdatasetdogcatdataset)为示例，train_list.txt/test_list.txt/validate_list.txt内容如下示例
```
cat/3270.jpg 0
cat/646.jpg 0
dog/12488.jpg 1
```

label_list.txt内容如下：
```
cat
dog
```


### 自定义数据加载

加载图像类自定义数据集，用户仅需要继承基类BaseCVDatast，修改数据集存放地址即可。具体使用如下：

**NOTE:**
* 数据集文件编码格式建议为utf8格式。
* dataset_dir为数据集实际路径，需要填写全路径，以下示例以`/test/data`为例。
* 训练/验证/测试集的数据列表文件中的图片路径需要相对于dataset_dir的相对路径，例如图片的实际位置为`/test/data/dog/dog1.jpg`。base_path为`/test/data`，则文件中填写的路径应该为`dog/dog1.jpg`。
* 如果您还有预测数据（没有文本类别），可以将预测数据存放在predict_list.txt文件，文件格式和train_list.txt类似。去掉label一列即可
* 如果您的数据集类别较少，可以不用定义label_list.txt，可以选择定义label_list=["数据集所有类别"]。
* 分类任务中，数据集的label必须从0开始计数

 ```python
from paddlehub.dataset.base_cv_dataset import BaseCVDataset

class DemoDataset(BaseCVDataset):
    def __init__(self):
        # 数据集存放位置
        self.dataset_dir = "/test/data"
        super(DemoDataset, self).__init__(
            base_path=self.dataset_dir,
            train_list_file="train_list.txt",
            validate_list_file="validate_list.txt",
            test_list_file="test_list.txt",
            predict_file="predict_list.txt",
            label_list_file="label_list.txt",
            # label_list=["数据集所有类别"])
dataset = DemoDataset()
```
