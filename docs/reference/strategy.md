# Strategy

在PaddleHub中，Strategy类封装了一系列适用于迁移学习的Fine-tune策略。Strategy包含了对预训练参数使用什么学习率变化策略，使用哪种类型的优化器，使用什么类型的正则化等。

### Class `hub.finetune.strategy.AdamWeightDecayStrategy`

```python
hub.AdamWeightDecayStrategy(
    learning_rate=1e-4,
    lr_scheduler="linear_decay",
    warmup_proportion=0.0,
    weight_decay=0.01,
    optimizer_name="adam")
```

基于Adam优化器的学习率衰减策略

**参数**

* learning_rate: 全局学习率，默认为1e-4
* lr_scheduler: 学习率调度方法，默认为"linear_decay"
* warmup_proportion: warmup所占比重
* weight_decay: 学习率衰减率
* optimizer_name: 优化器名称，默认为adam

**返回**

`AdamWeightDecayStrategy`

**示例**

```python
...
strategy = hub.AdamWeightDecayStrategy()

config = hub.RunConfig(
    use_cuda=True,
    num_epoch=10,
    batch_size=32,
    checkpoint_dir="hub_finetune_ckpt",
    strategy=strategy)
```

### Class `hub.finetune.strategy.DefaultFinetuneStrategy`

```python
hub.DefaultFinetuneStrategy(
    learning_rate=1e-4,
    optimizer_name="adam",
    regularization_coeff=1e-3)
```

默认的Finetune策略，该策略会对预训练参数增加L2正则作为惩罚因子

**参数**

* learning_rate: 全局学习率。默认为1e-4
* optimizer_name: 优化器名称。默认adam
* regularization_coeff: 正则化的λ参数。默认为1e-3

**返回**

`DefaultFinetuneStrategy`

**示例**

```python
...
strategy = hub.DefaultFinetuneStrategy()

config = hub.RunConfig(
    use_cuda=True,
    num_epoch=10,
    batch_size=32,
    checkpoint_dir="hub_finetune_ckpt",
    strategy=strategy)
```

### Class `hub.finetune.strategy.L2SPFinetuneStrategy`

```python
hub.L2SPFinetuneStrategy(
    learning_rate=1e-4,
    optimizer_name="adam",
    regularization_coeff=1e-3)
```

使用L2SP正则作为惩罚因子的Finetune策略

**参数**

* learning_rate: 全局学习率。默认为1e-4
* optimizer_name: 优化器名称。默认为adam
* regularization_coeff: 正则化的λ参数。默认为1e-3

**返回**

`L2SPFinetuneStrategy`

**示例**

```python
...
strategy = hub.L2SPFinetuneStrategy()

config = hub.RunConfig(
    use_cuda=True,
    num_epoch=10,
    batch_size=32,
    checkpoint_dir="hub_finetune_ckpt",
    strategy=strategy)
```

### Class `hub.finetune.strategy.ULMFiTStrategy`

```python
hub.ULMFiTStrategy(
    learning_rate=1e-4,
    optimizer_name="adam",
    cut_fraction=0.1,
  	ratio=32,
  	dis_blocks=3,
  	factor=2.6,
  	frz_blocks=3)
```

该策略实现了[ULMFiT](https://arxiv.org/abs/1801.06146)论文中提出的三种策略：Slanted triangular learning rates, Discriminative fine-tuning, Gradual unfreezing。

- Slanted triangular learning rates是一种学习率先上升再下降的策略,如下图所示：
<div align=center><img src="https://github.com/PaddlePaddle/PaddleHub/wiki/images/slanted.png" width="50%" height="50%"></div>

- Discriminative fine-tuning是一种学习率逐层递减的策略，通过该策略可以减缓底层的更新速度。
- Gradual unfreezing是一种逐层解冻的策略，通过该策略可以优先更新上层，再慢慢解冻下层参与更新。

**参数**

* learning_rate: 全局学习率。默认为1e-4。
* optimizer_name: 优化器名称。默认为adam。
* cut_fraction: 设置Slanted triangular learning rates学习率上升的步数在整个训练总步数中的比例，对应论文中Slanted triangular learning rates中的cut_frac。默认为0.1，如果设置为0，则不采用Slanted triangular learning rates。
* ratio: 设置Slanted triangular learning rates下降的最小学习率与上升的最大学习率的比例关系，默认为32，表示最小学习率是最大学习率的1/32。
* dis_blocks: 设置 Discriminative fine-tuning中的块数。由于预训练模型中没有记录op的层数，Paddlehub通过op的前后关系推测op所在的层次，这会导致诸如LSTM这类计算单元的op会被当作是不同层的op。为了不使层次划分太细，我们将层次进行了分块，用块的概念代替原论文中层的概念，通过设置dis_blocks即可设置块的个数。默认为3，如果设置为0，则不采用Discriminative fine-tuning。
* factor: 设置Discriminative fine-tuning的衰减率。默认为2.6，表示下一层的学习率是上一层的1/2.6。
* frz_blocks: 设置Gradual unfreezing中的块数。块的概念同“dis_blocks”中介绍的概念。

**返回**

`ULMFiTStrategy`

**示例**

```python
...
strategy = hub.ULMFiTStrategy()

config = hub.RunConfig(
    use_cuda=True,
    num_epoch=10,
    batch_size=32,
    checkpoint_dir="hub_finetune_ckpt",
    strategy=strategy)
```

### Class `hub.finetune.strategy.CombinedStrategy`

```python
hub.CombinedStrategy(
    optimizer_name="adam",
    learning_rate=1e-4,
    scheduler=None,
    regularization=None,
    clip=None)
```

Paddlehub中的基类策略，上文的所有策略都基于该策略，通过该策略可以设置所有策略参数。

**参数**

* optimizer_name: 优化器名称，默认为adam。
* learning_rate: 全局学习率，默认为1e-4。
* scheduler: 学习率调度方法，默认为None，此时不改变任何默认学习率调度方法参数，不采取任何学习率调度方法，即：
```python
scheduler = {
    "warmup": 0.0,
    "linear_decay": {
        "start_point": 1.0,
        "end_learning_rate": 0.0,
    },
    "noam_decay": False,
    "discriminative": {
        "blocks": 0,
        "factor": 2.6
    },
    "gradual_unfreeze": 0,
    "slanted_triangle": {
        "cut_fraction": 0.0,
        "ratio": 32
    }
}
```
* regularization: 正则方法，默认为None，此时不改变任何默认正则方法参数，不采取任何正则方法，即：

```python
regularization = {
    "L2": 0.0,
    "L2SP": 0.0,
    "weight_decay": 0.0,
}
```
* clip: 梯度裁剪方法，默认为None，此时不改变任何默认正则方法参数，不采取任何梯度裁剪方法，即：
```python
clip = {
    "GlobalNorm": 0.0,
    "Norm": 0.0
}
```

**返回**

`CombinedStrategy`

**示例**

```python
...
# Parameters not specified will remain default
scheduler = {
  "discriminative": {
    "blocks": 3,
    "factor": 2.6
  }
}
# Parameters not specified will remain default
regularization = {"L2": 1e-3}
# Parameters not specified will remain default
clip = {"GlobalNorm": 1.0}
strategy = hub.CombinedStrategy(
  scheduler = scheduler,
	regularization = regularization,
	clip = clip
)

config = hub.RunConfig(
    use_cuda=True,
    num_epoch=10,
    batch_size=32,
    checkpoint_dir="hub_finetune_ckpt",
    strategy=strategy)
```
