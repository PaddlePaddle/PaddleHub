# PaddleHub 超参优化（Auto Fine-tune）

## 一、简介

机器学习训练模型的过程中自然少不了调参。模型的参数可分成两类：参数与超参数，前者是模型通过自身的训练学习得到的参数数据；后者则需要通过人工经验设置（如学习率、dropout_rate、batch_size等），以提高模型训练的效果。当前模型往往参数空间大，手动调参十分耗时，尝试成本高。PaddleHub  Auto Fine-tune可以实现自动调整超参数。

PaddleHub Auto Fine-tune提供两种超参优化策略：

* HAZero: 核心思想是通过对正态分布中协方差矩阵的调整来处理变量之间的依赖关系和scaling。算法基本可以分成以下三步: 采样产生新解；计算目标函数值；更新正态分布参数。调整参数的基本思路为，调整参数使得产生好解的概率逐渐增大

* PSHE2: 采用粒子群算法，最优超参数组合就是所求问题的解。现在想求得最优解就是要找到更新超参数组合，即如何更新超参数，才能让算法更快更好的收敛到最优解。PSE2算法根据超参数本身历史的最优，在一定随机扰动的情况下决定下一步的更新方向。


PaddleHub Auto Fine-tune提供两种超参评估策略：

* FullTrail: 给定一组超参，利用这组超参从头开始Finetune一个新模型，之后在数据集dev部分评估这个模型

* ModelBased: 给定一组超参，若这组超参来自第一轮优化的超参，则从头开始Finetune一个新模型；若这组超参数不是来自第一轮优化的超参数，则程序会加载前几轮已经Fine-tune完毕后保存的较好模型，基于这个模型，在当前的超参数组合下继续Finetune。这个Fine-tune完毕后保存的较好模型，评估方式是这个模型在数据集dev部分的效果。

## 二、准备工作

使用PaddleHub Auto Fine-tune必须准备两个文件，并且这两个文件需要按照指定的格式书写。这两个文件分别是需要Fine-tune的python脚本finetuee.py和需要优化的超参数信息yaml文件hparam.yaml。

以Fine-tune中文情感分类任务为例，我们展示如何利用PaddleHub Auto Finetune进行超参优化。

以下是待优化超参数的yaml文件hparam.yaml，包含需要搜素的超参名字、类型、范围等信息。其中类型只支持float和int类型
```
param_list:
- name : learning_rate
  init_value : 0.001
  type : float
  lower_than : 0.05
  greater_than : 0.000005
- name : weight_decay
  init_value : 0.1
  type : float
  lower_than : 1
  greater_than : 0.0
- name : batch_size
  init_value : 32
  type : int
  lower_than : 40
  greater_than : 30
- name : warmup_prop
  init_value : 0.1
  type : float
  lower_than : 0.2
  greater_than : 0.0
```

**NOTE:** 该yaml文件的最外层级的key必须是param_list


以下是中文情感分类的finetunee.py

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast

import paddle.fluid as fluid
import paddlehub as hub
import os
from paddlehub.common.logger import logger

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=3, help="epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning_rate.")
parser.add_argument("--warmup_prop", type=float, default=0.1, help="warmup_prop.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay.")
parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--model_path", type=str, default="", help="load model path")
args = parser.parse_args()
# yapf: enable.


if __name__ == '__main__':
    # Load Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use ClassifyReader to read dataset
    dataset = hub.dataset.ChnSentiCorp()
    metrics_choices = ["acc"]

    reader = hub.reader.ClassifyReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    pooled_output = outputs["pooled_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of ERNIE's module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        warmup_proportion=args.warmup_prop,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler="linear_decay")

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        checkpoint_dir=args.checkpoint_dir,
        use_cuda=True,
        num_epoch=args.epochs,
        batch_size=args.batch_size,
        enable_memory_optim=True,
        strategy=strategy)

    # Define a classfication finetune task by PaddleHub's API
    cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=metrics_choices)

    # Finetune and evaluate by PaddleHub's API
    if args.model_path != "":
        with cls_task.phase_guard(phase="train"):
            cls_task.init_if_necessary()
            cls_task.load_parameters(args.model_path)
            logger.info("PaddleHub has loaded model from %s" % args.model_path)

    cls_task.finetune()
    run_states = cls_task.eval()
    eval_avg_score, eval_avg_loss, eval_run_speed = cls_task._calculate_metrics(run_states)

print(eval_avg_score["acc"], end="")
```
**Note**:以上是finetunee.py的写法。
> finetunee.py必须可以接收待优化超参数选项参数, 并且待搜素超参数选项名字和yaml文件中的超参数名字保持一致.

> finetunee.py必须有checkpoint_dir这个选项。

> PaddleHub Auto Fine-tune超参评估策略选择为ModelBased，finetunee.py必须有model_path选项。

> PaddleHub Auto Fine-tune优化超参策略选择hazero时，必须提供两个以上的待优化超参。

> finetunee.py的最后一个输出必须是模型在数据集dev上的评价效果，同时以“”结束，如print(eval_avg_score["acc"], end="").



## 三、启动方式

**确认安装PaddleHub版本在1.2.0以上, 同时PaddleHub Auto Fine-tune功能要求至少有一张GPU显卡可用。**

通过以下命令方式：
```shell
$ OUTPUT=result/
$ hub autofinetune finetunee.py --param_file=hparam.yaml --cuda=['1','2'] --popsize=5 --round=10
$ --output_dir=${OUTPUT} --evaluate_choice=fulltrail --tuning_strategy=hazero
```

其中，选项

> `--param_file`: 需要优化的超参数信息yaml文件

> `--cuda`: 设置运行程序的可用GPU卡号，list类型，中间以逗号隔开，不能有空格，默认为[‘0’]

> `--popsize`: 设置程序运行每轮产生的超参组合数，默认为5

> `--round`: 设置程序运行的轮数，默认是10

> `--output_dir`: 设置程序运行输出结果存放目录，可选，不指定该选项参数时，在当前运行路径下生成存放程序运行输出信息的文件夹

> `--evaluate_choice`: 设置自动优化超参的评价效果方式，可选fulltrail和modelbased, 默认为fulltrail

> `--tuning_strategy`: 设置自动优化超参策略，可选hazero和pshe2，默认为hazero

**NOTE:** Auto Fine-tune功能会根据popsize和cuda自动实现排队使用GPU，如popsize=5，cuda=['0','1','2','3']，则每搜索一轮，Auto Fine-tune自动起四个进程训练，所以第5组超参组合需要排队一次。为了提高GPU利用率以及超参优化效率，此时建议可以设置为3张可用的卡，cuda=['0','1','2']。


## 四、可视化

Auto Finetune API在优化超参过程中会自动对关键训练指标进行打点，启动程序后执行下面命令

```shell
$ tensorboard --logdir $OUTPUT/tb_paddle --host ${HOST_IP} --port ${PORT_NUM}
```

其中${HOST_IP}为本机IP地址，${PORT_NUM}为可用端口号，如本机IP地址为192.168.0.1，端口号8040，
用浏览器打开192.168.0.1:8040，即可看到搜素过程中各超参以及指标的变化情况

## 五、其他

如在使用Auto Fine-tune功能时，输出信息中包含如下字样：

**WARNING：Program which was ran with hyperparameters as ... was crashed!**

首先根据终端上的输出信息，确定这个输出信息是在第几个round（如round 3），之后查看${OUTPUT}/round3/下的日志文件信息log.info, 查看具体出错原因。
