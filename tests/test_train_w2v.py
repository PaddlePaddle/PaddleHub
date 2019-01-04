# coding=utf-8
from __future__ import print_function
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
import paddle_hub as hub
import unittest
import os

EMBED_SIZE = 64
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 1
PASS_NUM = 100

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)

_MOCK_DATA = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


def mock_data():
    for d in _MOCK_DATA:
        yield d


batch_reader = paddle.batch(mock_data, BATCH_SIZE)


def word2vec(words, is_sparse):
    embed_first = fluid.layers.embedding(
        input=words[0],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='embedding')
    embed_second = fluid.layers.embedding(
        input=words[1],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='embedding')
    embed_third = fluid.layers.embedding(
        input=words[2],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='embedding')
    embed_fourth = fluid.layers.embedding(
        input=words[3],
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='embedding')

    concat_emb = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_fourth], axis=1)
    hidden1 = fluid.layers.fc(input=concat_emb, size=HIDDEN_SIZE, act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')

    # declare later than predict word
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)

    return avg_cost


def train():
    place = fluid.CPUPlace()

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='fourthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    word_list = [first_word, second_word, third_word, forth_word, next_word]
    avg_cost = word2vec(word_list, is_sparse=True)

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=1e-3)

    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)
    exe.run(startup_program)  # initialization

    for epoch in range(0, PASS_NUM):
        for mini_batch in batch_reader():
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
            print("Cost = %f" % cost[0])

    model_dir = "./w2v_model"
    var_list_to_saved = [main_program.global_block().var("embedding")]
    print("saving model to %s" % model_dir)
    fluid.io.save_vars(
        executor=exe, dirname="./w2v_model/", vars=var_list_to_saved)


if __name__ == "__main__":
    train()
