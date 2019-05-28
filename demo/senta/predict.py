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
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for finetuning, input should be True or False")
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

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    with fluid.program_guard(program):
        sent_feature = outputs["sentence_feature"]

        # Define a classfication finetune task by PaddleHub's API
        cls_task = hub.create_text_cls_task(
            feature=sent_feature, num_classes=dataset.num_labels)

        # Setup feed list for data feeder
        # Must feed all the tensor of senta's module need
        feed_list = [inputs["words"].name, cls_task.variable('label').name]

        # classificatin probability tensor
        probs = cls_task.variable("probs")

        pred = fluid.layers.argmax(probs, axis=1)

        # load best model checkpoint
        fluid.io.load_persistables(exe, args.checkpoint_dir)

        inference_program = program.clone(for_test=True)

        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        test_reader = reader.data_generator(phase='test', shuffle=False)
        test_examples = dataset.get_test_examples()
        total = 0
        correct = 0
        for index, batch in enumerate(test_reader()):
            pred_v = exe.run(
                feed=data_feeder.feed(batch),
                fetch_list=[pred.name],
                program=inference_program)
            total += 1
            if (pred_v[0][0] == int(test_examples[index].label)):
                correct += 1
                acc = 1.0 * correct / total
            print("%s\tpredict=%s" % (test_examples[index], pred_v[0][0]))
    print("accuracy = %f" % acc)
