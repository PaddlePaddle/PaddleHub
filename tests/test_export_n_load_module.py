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

from __future__ import print_function
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
import paddle
import paddle_hub as hub
import unittest
import os

from collections import defaultdict

EMBED_SIZE = 16
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 64
PASS_NUM = 1000

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)
data = paddle.dataset.imikolov.train(word_dict, N)

_MOCK_DATA = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


def mock_data():
    for d in _MOCK_DATA:
        yield d


batch_reader = paddle.batch(mock_data, BATCH_SIZE)
#batch_reader = paddle.batch(data, BATCH_SIZE)
batch_size = 0
for d in batch_reader():
    batch_size += 1


def word2vec(words, is_sparse, trainable=True):
    emb_param_attr = fluid.ParamAttr(name="embedding", trainable=trainable)
    embed_first = fluid.layers.embedding(
        input=words[0],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=emb_param_attr)
    embed_second = fluid.layers.embedding(
        input=words[1],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=emb_param_attr)
    embed_third = fluid.layers.embedding(
        input=words[2],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=emb_param_attr)
    embed_fourth = fluid.layers.embedding(
        input=words[3],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=emb_param_attr)

    concat_emb = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(input=concat_emb, size=HIDDEN_SIZE, act='sigmoid')
    pred_prob = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')

    # declare later than predict word
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=pred_prob, label=next_word)
    avg_cost = fluid.layers.mean(cost)

    return pred_prob, avg_cost


def get_dictionary(word_dict):
    dictionary = defaultdict(int)
    w_id = 0
    for w in word_dict:
        if isinstance(w, bytes):
            w = w.decode("ascii")
        dictionary[w] = w_id
        w_id += 1

    return dictionary


def test_create_w2v_module(use_gpu=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    word_list = [first_word, second_word, third_word, forth_word, next_word]
    pred_prob, avg_cost = word2vec(word_list, is_sparse=True)

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=1e-2)

    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)
    exe.run(startup_program)  # initialization

    step = 0
    for epoch in range(0, PASS_NUM):
        for mini_batch in batch_reader():
            feed_var_list = [
                main_program.global_block().var("firstw"),
                main_program.global_block().var("secondw"),
                main_program.global_block().var("thirdw"),
                main_program.global_block().var("fourthw"),
                main_program.global_block().var("nextw")
            ]
            feeder = fluid.DataFeeder(feed_list=feed_var_list, place=place)
            cost = exe.run(
                main_program,
                feed=feeder.feed(mini_batch),
                fetch_list=[avg_cost])
            step += 1
            if step % 100 == 0:
                print("Epoch={} Step={} Cost={}".format(epoch, step, cost[0]))

    saved_module_dir = "./tmp/word2vec_test_module"
    # save inference model including feed and fetch variable info
    dictionary = get_dictionary(word_dict)

    module_inputs = [
        main_program.global_block().var("firstw"),
        main_program.global_block().var("secondw"),
        main_program.global_block().var("thirdw"),
        main_program.global_block().var("fourthw"),
    ]
    signature = hub.create_signature(
        "default", inputs=module_inputs, outputs=[pred_prob])
    hub.create_module(
        sign_arr=signature,
        program=fluid.default_main_program(),
        module_dir=saved_module_dir,
        word_dict=dictionary)


def test_load_w2v_module(use_gpu=False):
    saved_module_dir = "./tmp/word2vec_test_module"
    w2v_module = hub.Module(module_dir=saved_module_dir)
    feed_list, fetch_list, program = w2v_module(
        sign_name="default", trainable=False)
    with fluid.program_guard(main_program=program):
        pred_prob = fetch_list[0]

        pred_word = fluid.layers.argmax(x=pred_prob, axis=1)
        # set place, executor, datafeeder
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(place=place, feed_list=feed_list)

        word_ids = [[1, 2, 3, 4]]
        result = exe.run(
            fluid.default_main_program(),
            feed=feeder.feed(word_ids),
            fetch_list=[pred_word],
            return_numpy=True)

        print(result)


if __name__ == "__main__":
    use_gpu = False
    print("test create word2vec module")
    test_create_w2v_module(use_gpu)
    print("test load word2vec module")
    test_load_w2v_module(use_gpu=False)
