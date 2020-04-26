# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def textcnn_net(data,
                dict_dim,
                emb_dim=128,
                hid_dim=128,
                hid_dim2=96,
                class_dim=3,
                win_sizes=None):
    """
    Textcnn_net
    """
    if win_sizes is None:
        win_sizes = [1, 2, 3]

    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    # convolution layer
    convs = []
    for win_size in win_sizes:
        conv_h = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max")
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    fc_1 = fluid.layers.fc(input=[convs_out], size=hid_dim2, act="tanh")
    # softmax layer
    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax")

    return prediction, fc_1
