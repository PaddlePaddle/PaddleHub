训练一个新任务时，如果从零开始训练时，这将是一个耗时的过程，并且效果可能达不到理想的效果，此时您可以利用PaddleHub提供的预训练模型进行具体任务的FineTune。您只需要对自定义数据进行相应的预处理，随后输入预训练模型中，即可得到相应的结果。本文以预训练模型ERNIE对文本分类任务进行FineTune为例，说明如何利用PaddleHub适配自定义数据完成FineTune。

# 数据准备

> * train.tsv 训练集
> * dev.tsv 验证集
> * test.tsv 测试集

相应的数据格式为第一列是文本的编号（guid），第二列为文本内容，第三列为另一文本内容，第四列为标签，列与列之间以Tab键分隔。
**NOTE:** 若是单文本分类任务，则第三列相应内容为空。

```
9566	挺无聊的一本书。内容很贴近生活，可能是因为太贴近生活了，反而没什么可看的了。	None	0
9529	内存数量配置偏低 内存插槽于掌托下，需拆卸安装，不方便 蓝牙模块采用软件控制	None	0
9544	这本书看得我实在是没什么感觉。作者说的很抽象，我感觉挺空洞的，没什么意思。有点后悔买这本书。	None	0
```

# 自定义数据预处理
在源码paddlehub/dataset下自定义demodataset.py，便于数据预处理。

**示例**
```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import codecs
import os
import csv

from paddlehub.dataset import InputExample, HubDataset

class DemoDataset(HubDataset):
    """DemoDataset"""
    def __init__(self):
        self.dataset_dir = "path/to/dataset"

        self._load_train_examples()
        self._load_test_examples()
        self._load_dev_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "train.tsv")
        self.train_examples = self._read_tsv(self.train_file)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "dev.tsv")
        self.dev_examples = self._read_tsv(self.dev_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "test.tsv")
        self.test_examples = self._read_tsv(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        """define it according the real dataset"""
        return ["0", "1"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[0], text_a=line[1])
                seq_id += 1
                examples.append(example)

            return examples
```
***

之后，您就可以通过hub.dataset.DemoDataset()获取自定义数据集了。进而配合ClassifyReader以及预训练模型如ERNIE完成文本分类任务。
