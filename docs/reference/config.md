## hub.config

在PaddleHub中，RunConfig代表了在对[Task](./task)进行Fine-tune时的运行配置。包括运行的epoch次数、batch的大小、是否使用GPU训练等。

### Class `hub.finetune.config.RunConfig`

```python
hub.RunConfig(
    log_interval=10,
    eval_interval=100,
    use_pyreader=True,
    use_data_parallel=True,
    save_ckpt_interval=None,
    use_cuda=False,
    checkpoint_dir=None,
    num_epoch=10,
    batch_size=None,
    enable_memory_optim=False,
    strategy=None)`
```
**参数:**

* `log_interval`: 打印训练日志的周期，默认为10。
* `eval_interval`: 进行评估的周期，默认为100。
* `use_pyreader`: 是否使用pyreader，默认True。
* `use_data_parallel`: 是否使用并行计算，默认True。打开该功能依赖nccl库。
* `save_ckpt_interval`: 保存checkpoint的周期，默认为None。
* `use_cuda`: 是否使用GPU训练和评估，默认为False。
* `checkpoint_dir`: checkpoint的保存目录，默认为None，此时会在工作目录下根据时间戳生成一个临时目录。
* `num_epoch`: 运行的epoch次数，默认为10。
* `batch_size`: batch大小，默认为None。
* `enable_memory_optim`: 是否进行内存优化，默认为False。
* `strategy`: finetune的策略。默认为None，此时会使用DefaultFinetuneStrategy策略。

**返回**

`RunConfig`

**示例**

```python
import paddlehub as hub

config = hub.RunConfig(
    use_cuda=True,
    num_epoch=10,
    batch_size=32)
```

#### `log_interval`

获取RunConfig设置的log_interval属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
log_interval = config.log_interval()
```

#### `eval_interval`

获取RunConfig设置的eval_interval属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
eval_interval = config.eval_interval()
```

#### `use_pyreader`

获取RunConfig设置的use_pyreader属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
use_pyreader = config.use_pyreader()
```

#### `use_data_parallel`

获取RunConfig设置的use_data_parallel属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
use_data_parallel = config.use_data_parallel()
```

#### `save_ckpt_interval`

获取RunConfig设置的save_ckpt_interval属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
save_ckpt_interval = config.save_ckpt_interval()
```

#### `use_cuda`

获取RunConfig设置的use_cuda属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
use_cuda = config.use_cuda()
```

#### `checkpoint_dir`

获取RunConfig设置的checkpoint_dir属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
checkpoint_dir = config.checkpoint_dir()
```

#### `num_epoch`

获取RunConfig设置的num_epoch属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
num_epoch = config.num_epoch()
```

#### `batch_size`

获取RunConfig设置的batch_size属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
batch_size = config.batch_size()
```

#### `strategy`

获取RunConfig设置的strategy属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
strategy = config.strategy()
```

#### `enable_memory_optim`

获取RunConfig设置的enable_memory_optim属性

**示例**

```python
import paddlehub as hub

config = hub.RunConfig()
enable_memory_optim = config.enable_memory_optim()
```
