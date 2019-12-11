#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import numpy as np
import os
import time

import paddle
import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str,                 default=None, help="Directory to model checkpoint")
parser.add_argument("--use_gpu",        type=ast.literal_eval,    default=True, help="Whether use GPU for finetuning, input should be True or False")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # loading Paddlehub senta pretrained model
    module = hub.Module(name="senta_bilstm")
    inputs, outputs, program = module.context(trainable=True)

    # Sentence classification  dataset reader
    dataset = hub.dataset.ChnSentiCorp()
    reader = hub.reader.LACClassifyReader(
        dataset=dataset, vocab_path=module.get_vocab_path())

    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=0.01,
        warmup_proportion=0.1,
        learning_rate=5e-5,
        lr_scheduler="linear_decay",
        optimizer_name="adam")

    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=False,
        use_cuda=args.use_gpu,
        batch_size=1,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    sent_feature = outputs["sentence_feature"]

    feed_list = [inputs["words"].name]

    cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=sent_feature,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config)

    data = ["这家餐厅很好吃", "这部电影真的很差劲"]

    run_states = cls_task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    index = 0
    for batch_result in results:
        batch_result = np.argmax(batch_result, axis=2)[0]
        for result in batch_result:
            print("%s\tpredict=%s" % (data[index], result))
            index += 1
