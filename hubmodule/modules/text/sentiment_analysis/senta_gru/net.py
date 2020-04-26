# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def gru_net(data,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            emb_lr=30.0):
    """
    gru net
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 3)

    # GRU layer
    gru_h = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)
    gru_max = fluid.layers.sequence_pool(input=gru_h, pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max)

    # full connect layer
    fc1 = fluid.layers.fc(input=gru_max_tanh, size=hid_dim2, act='tanh')
    # softmax layer
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    return prediction, fc1
