# Strategy
----
在PaddleHub中，Strategy类封装了一系列适用于迁移学习的Fine-tuning策略。Strategy包含了对预训练参数使用什么学习率变化策略，使用哪种类型的优化器，使用什么类型的正则化等。

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
> * learning_rate: 全局学习率，默认为1e-4
> * lr_scheduler: 学习率调度方法，默认为"linear_decay"
> * warmup_proportion: warmup所占比重
> * weight_decay: 学习率衰减率
> * optimizer_name: 优化器名称，默认为None，此时会使用Adam

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
> * learning_rate: 全局学习率。默认为1e-4
> * optimizer_name: 优化器名称。默认为None，此时会使用Adam
> * regularization_coeff: 正则化的λ参数。默认为1e-3

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
> * learning_rate: 全局学习率。默认为1e-4
> * optimizer_name: 优化器名称。默认为None，此时会使用Adam
> * regularization_coeff: 正则化的λ参数。默认为1e-3

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
