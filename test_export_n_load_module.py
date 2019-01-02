# coding: utf-8
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


#batch_reader = paddle.batch(mock_data, BATCH_SIZE)
batch_reader = paddle.batch(data, BATCH_SIZE)
batch_size = 0
for d in batch_reader():
    batch_size += 1
print("imikolov simple dataset batch_size =", batch_size)


def module_fn(trainable=False):
    # create word input
    words = fluid.layers.data(
        name="words", shape=[1], lod_level=1, dtype="int64")

    # create embedding
    # emb_name = "{}:embedding".format(module_scope)
    emb_name = "embedding"
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

    # fluid.io.load_vars(
    #     executor=exe,
    #     dirname="./w2v_model",
    #     vars=[main_program.global_block().var("embedding")])
    # 也可使用predicate方式搜索变量
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

    model_dir = "./w2v_model"
    # save part of model
    var_list_to_saved = [main_program.global_block().var("embedding")]
    print("saving model to %s" % model_dir)
    fluid.io.save_vars(
        executor=exe, dirname=model_dir + "_save_vars", vars=var_list_to_saved)

    # save the whole model
    fluid.io.save_persistables(
        executor=exe, dirname=model_dir + "_save_persistables")

    saved_model_path = "w2v_saved_inference_model"
    # save inference model including feed and fetch variable info
    fluid.io.save_inference_model(
        dirname=saved_model_path,
        feeded_var_names=["firstw", "secondw", "thirdw", "fourthw"],
        target_vars=[predict_word],
        executor=exe)

    dictionary = []
    for w in word_dict:
        if isinstance(w, bytes):
            w = w.decode("ascii")
        dictionary.append(w)

    # save word dict to assets folder
    hub.ModuleConfig.save_module_dict(
        module_path=saved_model_path, word_dict=dictionary)


def test_save_module(use_cuda=False):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)
    main_program = fluid.Program()
    startup_program = fluid.Program()
    exe = fluid.Executor(place)

    with fluid.program_guard(main_program, startup_program):
        words, word_emb = module_fn()
        exe.run(startup_program)
        # load inference embedding parameters
        saved_model_path = "./w2v_saved_inference_model"
        fluid.io.load_inference_model(executor=exe, dirname=saved_model_path)

        feed_var_list = [main_program.global_block().var("words")]
        feeder = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        results = exe.run(
            main_program,
            feed=feeder.feed([[[1, 2, 3, 4, 5]]]),
            fetch_list=[word_emb],
            return_numpy=False)

        np_result = np.array(results[0])
        print(np_result)

        saved_module_dir = "./test/word2vec_inference_module"
        fluid.io.save_inference_model(
            dirname=saved_module_dir,
            feeded_var_names=["words"],
            target_vars=[word_emb],
            executor=exe)

        dictionary = []
        for w in word_dict:
            if isinstance(w, bytes):
                w = w.decode("ascii")
            dictionary.append(w)
        # save word dict to assets folder
        config = hub.ModuleConfig(saved_module_dir)
        config.save_dict(word_dict=dictionary)

        config.dump()


def test_load_module(use_cuda=False):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(fluid.CPUPlace())
    saved_module_dir = "./test/word2vec_inference_module"
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(
         saved_module_dir, executor=exe)

    # Sequence input in Paddle must be LOD Tensor, so we need to convert them inside Module
    word_ids = [[1, 2, 3, 4, 5]]
    lod = [[5]]
    word_ids_lod_tensor = fluid.create_lod_tensor(word_ids, lod, place)

    results = exe.run(
        inference_program,
        feed={feed_target_names[0]: word_ids_lod_tensor},
        fetch_list=fetch_targets,
        return_numpy=False)

    print(feed_target_names)
    print(fetch_targets)
    np_result = np.array(results[0])
    print(np_result)


if __name__ == "__main__":
    use_cuda = True
    train(use_cuda)
    test_save_module()
    test_load_module()
