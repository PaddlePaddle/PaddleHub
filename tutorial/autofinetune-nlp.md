# PaddleHub 超参优化（Auto Fine-tune）——NLP情感分类任务

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
parser.add_argument("--saved_params_dir", type=str, default="", help="Directory for saving model during ")
parser.add_argument("--model_path", type=str, default="", help="load model path")
args = parser.parse_args()
# yapf: enable.


def is_path_valid(path):
    if path == "":
        return False
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return True

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

    # Load model from the defined model path or not
    if args.model_path != "":
        with cls_task.phase_guard(phase="train"):
            cls_task.init_if_necessary()
            cls_task.load_parameters(args.model_path)
            logger.info("PaddleHub has loaded model from %s" % args.model_path)

    cls_task.finetune()
    run_states = cls_task.eval()
    eval_avg_score, eval_avg_loss, eval_run_speed = cls_task._calculate_metrics(run_states)

    # Move ckpt/best_model to the defined saved parameters directory
    if is_path_valid(args.saved_params_dir) and os.path.exists(config.checkpoint_dir+"/best_model/"):
        shutil.copytree(config.checkpoint_dir+"/best_model/", args.saved_params_dir)
        shutil.rmtree(config.checkpoint_dir)

    print("AutoFinetuneEval"+"\t"+str(float(eval_avg_score["acc"])))
```
**Note**:以上是finetunee.py的写法。
> finetunee.py必须可以接收待优化超参数选项参数, 并且待搜素超参数选项名字和yaml文件中的超参数名字保持一致。

> finetunee.py必须有saved_params_dir这个选项。

> PaddleHub Auto Fine-tune超参评估策略选择为ModelBased，finetunee.py必须有model_path选项。

> PaddleHub Auto Fine-tune优化超参策略选择hazero时，必须提供两个以上的待优化超参。

> finetunee.py必须输出模型在数据集dev上的评价效果，同时以“AutoFinetuneEval"开始，和评价效果之间以“\t”分开，如print("AutoFinetuneEval"+"\t"+str(float(eval_avg_score["acc"])))。
