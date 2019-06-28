PaddleHub提供以下数据集可供下载：

### Class `hub.dataset.ChnSentiCorp`
ChnSentiCorp 是中文情感分析数据集，其目标是判断一段话的情感态度。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.ChnSentiCorp()
> ```

### Class `hub.dataset.LCQMC`
LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问答匹配数据集，其目标是判断两个问题的语义是否相同。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.LCQMC()
> ```

### Class `hub.dataset.NLPCC_DPQA`
NLPCC_DPQA 是由国际自然语言处理和中文计算会议NLPCC于2016年举办的评测任务，其目标是选择能够回答问题的答案。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.NLPCC_DPQA()
> ```

### Class `hub.dataset.MSRA_NER`
MSRA-NER(SIGHAN 2006) 数据集由微软亚研院发布，其目标是命名实体识别，是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名等。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.MSRA-NER()
> ```

### Class `hub.dataset.DogCatDataset`
DOGCAT 是由Kaggle提供的数据集，用于图像二分类，其目标是判断一张图片是猫或是狗。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.DogCatDataset()
> ```

### Class `hub.dataset.Food101Dataset`
FOOD101 是由Kaggle提供的数据集，含有101种类别，其目标是判断一张图片属于101种类别中的哪一种类别。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.Food101Dataset()
> ```

### Class `hub.dataset.Indoor67Dataset`
INDOOR数据集是由麻省理工学院发布，其包含67种室内场景，其目标是识别一张室内图片的场景类别。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.Indoor67Dataset()
> ```

### Class `hub.dataset.FlowersDataset`
FLOWERS数据集是是公开花卉数据集，一共有5种类型，用于做图像分类。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.FlowersDataset()
> ```

### Class `hub.dataset.StanfordDogsDataset`
STANFORD_DOGS数据集是斯坦福大学发布，其包含120个种类的狗，用于做图像分类。

**示例**
>
> ```python
> import paddlehub as hub
>
>dataset = hub.dataset.StanfordDogsDataset()
> ```


若您想在自定义数据集上完成FineTune，请查看[PaddleHub适配自定义数据完成FineTune](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/turtorial/user_define_dataset.md)
