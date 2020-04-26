# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def lstm_net(data,
             dict_dim,
             emb_dim=128,
             hid_dim=128,
             hid_dim2=96,
             class_dim=2,
             emb_lr=30.0):
    """
    Lstm net
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    # Lstm layer
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)
    # max pooling layer
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)
    # full connect layer
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')
    # softmax layer
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')

    return prediction, fc1
