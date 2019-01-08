#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
import sys
import os
import time
import unittest
import contextlib
import logging
import argparse
import ast
import utils

import paddle.fluid as fluid
import paddle_hub as hub

from nets import bow_net
from nets import cnn_net
from nets import lstm_net
from nets import bilstm_net
from nets import gru_net
logger = logging.getLogger("paddle-fluid")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("Sentiment Classification.")
    # training data path
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=False,
        help="The path of trainning data. Should be given in train mode!")
    # test data path
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=False,
        help="The path of test data. Should be given in eval or infer mode!")
    # word_dict path
    parser.add_argument(
        "--word_dict_path",
        type=str,
        required=True,
        help="The path of word dictionary.")
    # current mode
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['train', 'eval', 'infer'],
        help="train/eval/infer mode")
    # model type
    parser.add_argument(
        "--model_type", type=str, default="bow_net", help="type of model")
    # model save path parser.add_argument(
    parser.add_argument(
        "--model_path",
        type=str,
        default="models",
        required=True,
        help="The path to saved the trained models.")
    # Number of passes for the training task.
    parser.add_argument(
        "--num_passes",
        type=int,
        default=3,
        help="Number of passes for the training task.")
    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The number of training examples in one forward/backward pass.")
    # lr value for training
    parser.add_argument(
        "--lr", type=float, default=0.002, help="The lr value for training.")
    # Whether to use gpu
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=False,
        help="Whether to use gpu to train the model.")
    # parallel train
    parser.add_argument(
        "--is_parallel",
        type=ast.literal_eval,
        default=False,
        help="Whether to train the model in parallel.")
    args = parser.parse_args()
    return args


def bow_net_module(data,
                   label,
                   dict_dim,
                   emb_dim=128,
                   hid_dim=128,
                   hid_dim2=96,
                   class_dim=2):
    """
    Bow net
    """
    module_dir = "./model/test_create_module"
    # embedding layer
    emb = fluid.layers.embedding(
        input=data, size=[dict_dim, emb_dim], param_attr="bow_embedding")
    # bow layer
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    # full connect layer
    fc_1 = fluid.layers.fc(
        input=bow_tanh, size=hid_dim, act="tanh", name="bow_fc1")
    fc_2 = fluid.layers.fc(
        input=fc_1, size=hid_dim2, act="tanh", name="bow_fc2")
    # softmax layer
    prediction = fluid.layers.fc(
        input=[fc_2], size=class_dim, act="softmax", name="fc_softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction, emb


def train_net(train_reader,
              word_dict,
              network_name,
              use_gpu,
              parallel,
              save_dirname,
              lr=0.002,
              batch_size=128,
              pass_num=10):
    """
    train network
    """
    if network_name == "bilstm_net":
        network = bilstm_net
    elif network_name == "bow_net":
        network = bow_net
    elif network_name == "cnn_net":
        network = cnn_net
    elif network_name == "lstm_net":
        network = lstm_net
    elif network_name == "gru_net":
        network = gru_net
    else:
        print("unknown network type")
        return

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    cost, acc, pred, emb = network(data, label, len(word_dict) + 2)

    # set optimizer
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    # set place, executor, datafeeder
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=["words", "label"], place=place)
    exe.run(fluid.default_startup_program())
    # start training...

    for pass_id in range(pass_num):
        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        for batch in train_reader():
            avg_cost_np, avg_acc_np = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(batch),
                fetch_list=[cost, acc],
                return_numpy=True)
            data_size = len(batch)
            total_acc += data_size * avg_acc_np
            total_cost += data_size * avg_cost_np
            data_count += data_size
        avg_cost = total_cost / data_count
        avg_acc = total_acc / data_count
        print("[train info]: pass_id: %d, avg_acc: %f, avg_cost: %f" %
              (pass_id, avg_acc, avg_cost))

    # save the model
    module_dir = os.path.join(save_dirname, network_name)
    config = hub.ModuleConfig(module_dir)
    config.save_dict(word_dict=word_dict)

    # saving config
    input_desc = {"words": data.name}
    output_desc = {"emb": emb.name}
    config.register_feed_signature(input_desc)
    config.register_fetch_signature(output_desc)
    config.dump()
    feed_var_name = config.feed_var_name("words")
    fluid.io.save_inference_model(module_dir, [feed_var_name], emb, exe)


def retrain_net(train_reader,
                word_dict,
                network_name,
                use_gpu,
                parallel,
                save_dirname,
                lr=0.002,
                batch_size=128,
                pass_num=30):
    """
    train network
    """
    if network_name == "bilstm_net":
        network = bilstm_net
    elif network_name == "bow_net":
        network = bow_net
    elif network_name == "cnn_net":
        network = cnn_net
    elif network_name == "lstm_net":
        network = lstm_net
    elif network_name == "gru_net":
        network = gru_net
    else:
        print("unknown network type")
        return

    dict_dim = len(word_dict) + 2
    emb_dim = 128
    hid_dim = 128
    hid_dim2 = 96
    class_dim = 2

    module_path = "./models/bow_net"
    module = hub.Module(module_dir=module_path)

    main_program = fluid.Program()
    startup_program = fluid.Program()

    # use switch program to test fine-tuning
    fluid.framework.switch_main_program(module.get_inference_program())

    # remove feed fetch operator and variable
    hub.ModuleUtils.remove_feed_fetch_op(fluid.default_main_program())

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    data = module.get_feed_var("words")
    emb = module.get_fetch_var("emb")

    # bow layer
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    # full connect layer
    fc_1 = fluid.layers.fc(
        input=bow_tanh, size=hid_dim, act="tanh", name="bow_fc1")
    fc_2 = fluid.layers.fc(
        input=fc_1, size=hid_dim2, act="tanh", name="bow_fc2")
    # softmax layer
    pred = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.mean(
        fluid.layers.cross_entropy(input=pred, label=label))
    acc = fluid.layers.accuracy(input=pred, label=label)

    # set optimizer
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    # set place, executor, datafeeder
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=["words", "label"], place=place)
    exe.run(fluid.default_startup_program())

    # start training...
    for pass_id in range(pass_num):
        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        for batch in train_reader():
            avg_cost_np, avg_acc_np = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(batch),
                fetch_list=[cost, acc],
                return_numpy=True)
            data_size = len(batch)
            total_acc += data_size * avg_acc_np
            total_cost += data_size * avg_cost_np
            data_count += data_size
        avg_cost = total_cost / data_count
        avg_acc = total_acc / data_count
        print("[train info]: pass_id: %d, avg_acc: %f, avg_cost: %f" %
              (pass_id, avg_acc, avg_cost))

    # save the model
    module_dir = os.path.join(save_dirname, network_name + "_retrain")
    fluid.io.save_inference_model(module_dir, ["words"], emb, exe)

    config = hub.ModuleConfig(module_dir)
    config.save_dict(word_dict=word_dict)
    config.dump()


def main(args):

    # prepare_data to get word_dict, train_reader
    word_dict, train_reader = utils.prepare_data(
        args.train_data_path, args.word_dict_path, args.batch_size, args.mode)

    train_net(train_reader, word_dict, args.model_type, args.use_gpu,
              args.is_parallel, args.model_path, args.lr, args.batch_size,
              args.num_passes)

    # NOTE(ZeyuChen): can't run train_net and retrain_net together
    # retrain_net(train_reader, word_dict, args.model_type, args.use_gpu,
    #             args.is_parallel, args.model_path, args.lr, args.batch_size,
    #             args.num_passes)


if __name__ == "__main__":
    args = parse_args()
    main(args)
