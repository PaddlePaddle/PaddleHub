# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def bow_net(emb, seq_len, hid_dim=128, hid_dim2=96, class_dim=2):
    """
    Bow net
    """
    # unpad the token_feature
    unpad_feature = fluid.layers.sequence_unpad(emb, length=seq_len)

    # bow layer
    bow = fluid.layers.sequence_pool(input=unpad_feature, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    # full connect layer
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")

    # softmax layer
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")

    return prediction, fc_2
