#-*- coding:utf8 -*-
import paddle
import paddle.fluid as fluid
import paddle_hub as hub
import paddle_hub.module as module
import paddle_hub.logger as log
import sys
import numpy as np
import reader
import argparse
import functools
from visualdl import LogWriter
from utility import add_arguments, print_arguments

reader = paddle.batch(reader.test("dataset"), batch_size=1)


def infer():

    model = module.Module(module_dir="hub_module_ResNet50")

    feed_list, fetch_list, program = model(
        sign_name="feature_map", trainable=True)

    with fluid.program_guard(main_program=program):
        img = feed_list[0]
        feature_map = fetch_list[0]
        fc = fluid.layers.fc(input=feature_map, size=2, act="softmax")
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[img], place=place)
        exe.run(fluid.default_startup_program())
        for batch in reader():
            print(batch[0][0].shape)
            eval_val = exe.run(fetch_list=[fc.name], feed=feeder.feed(batch))
            log.logger.info(eval_val)
            input()


infer()
