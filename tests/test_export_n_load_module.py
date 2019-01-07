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
PASS_NUM = 1

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
print("imikolov simple dataset batch_size =", batch_size)


def module_fn(trainable=False):
    # Define module function for saving module
    # create word input
    words = fluid.layers.data(
        name="words", shape=[1], lod_level=1, dtype="int64")

    # create embedding
    emb_name = "w2v_emb"
    emb_param_attr = fluid.ParamAttr(name=emb_name, trainable=trainable)
    word_emb = fluid.layers.embedding(
        input=words,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=True,
        param_attr=emb_param_attr)

    # return feeder and fetch_list
    return words, word_emb


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
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')

    # declare later than predict word
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)

    return predict_word, avg_cost


def train(use_cuda=False):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    word_list = [first_word, second_word, third_word, forth_word, next_word]
    predict_word, avg_cost = word2vec(word_list, is_sparse=True)

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=1e-3)

    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)
    exe.run(startup_program)  # initialization

    step = 0
    for epoch in range(0, PASS_NUM):
        for mini_batch in batch_reader():
            # print("mini_batch", mini_batch)
            # 定义输入变量
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

    saved_model_dir = "./tmp/word2vec_test_model"
    # save inference model including feed and fetch variable info
    fluid.io.save_inference_model(
        dirname=saved_model_dir,
        feeded_var_names=["firstw", "secondw", "thirdw", "fourthw"],
        target_vars=[predict_word],
        executor=exe)

    dictionary = defaultdict(int)
    w_id = 0
    for w in word_dict:
        if isinstance(w, bytes):
            w = w.decode("ascii")
        dictionary[w] = w_id
        w_id += 1

    # save word dict to assets folder
    config = hub.ModuleConfig(saved_model_dir)
    config.save_dict(word_dict=dictionary)
    config.dump()


def test_save_module(use_cuda=False):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    main_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(main_program, startup_program):
        words, word_emb = module_fn()
        exe.run(startup_program)
        # load inference embedding parameters
        saved_model_dir = "./tmp/word2vec_test_model"
        fluid.io.load_inference_model(executor=exe, dirname=saved_model_dir)

        # feed_var_list = [main_program.global_block().var("words")]
        # feeder = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        # results = exe.run(
        #     main_program,
        #     feed=feeder.feed([[[1, 2, 3, 4, 5]]]),
        #     fetch_list=[word_emb],
        #     return_numpy=False)

        # np_result = np.array(results[0])
        # print(np_result)

        # save module_dir
        saved_module_dir = "./tmp/word2vec_test_module"
        fluid.io.save_inference_model(
            dirname=saved_module_dir,
            feeded_var_names=["words"],
            target_vars=[word_emb],
            executor=exe)

        dictionary = defaultdict(int)
        w_id = 0
        for w in word_dict:
            if isinstance(w, bytes):
                w = w.decode("ascii")
            dictionary[w] = w_id
            w_id += 1

        signature = hub.create_signature(
            "default", inputs=[words], outputs=[word_emb])
        hub.create_module(
            sign_arr=signature, program=main_program, path=saved_module_dir)


def test_load_module(use_cuda=False):
    saved_module_dir = "./tmp/word2vec_test_module"
    w2v_module = hub.Module(module_dir=saved_module_dir)

    word_ids = [[1, 2, 3, 4, 5]]  # test sequence
    word_ids_lod_tensor = w2v_module._preprocess_input(word_ids)
    result = w2v_module({"words": word_ids_lod_tensor})
    print(result)


if __name__ == "__main__":
    use_cuda = False
    print("train...")
    train(use_cuda)
    print("save module...")
    test_save_module()
    print("load module...")
    test_load_module()
