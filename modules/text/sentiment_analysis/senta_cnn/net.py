# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def cnn_net(emb, seq_len, hid_dim=128, hid_dim2=96, class_dim=2, win_size=3):
    """
    Conv net
    """
    # unpad the token_feature
    unpad_feature = fluid.layers.sequence_unpad(emb, length=seq_len)

    # convolution layer
    conv_3 = fluid.nets.sequence_conv_pool(
        input=unpad_feature, num_filters=hid_dim, filter_size=win_size, act="tanh", pool_type="max")
    # full connect layer
    fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2)

    # softmax layer
    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")

    return prediction, fc_1
