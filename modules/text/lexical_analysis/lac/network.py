# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def lex_net(word_dict_len, label_dict_len):
    """
    define the lexical analysis network structure
    """
    word_emb_dim = 128
    grnn_hidden_dim = 128
    emb_lr = 2
    crf_lr = 0.2
    bigru_num = 2
    init_bound = 0.1
    IS_SPARSE = True

    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))

        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge

    def _net_conf(word):
        """
        Configure the network
        """
        word_embedding = fluid.layers.embedding(
            input=word,
            size=[word_dict_len, word_emb_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound)))

        input_feature = word_embedding
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        emission = fluid.layers.fc(
            size=label_dict_len,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))

        size = emission.shape[1]
        fluid.layers.create_parameter(shape=[size + 2, size], dtype=emission.dtype, name='crfw')
        crf_decode = fluid.layers.crf_decoding(input=emission, param_attr=fluid.ParamAttr(name='crfw'))

        return crf_decode, emission

    word = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)

    crf_decode, emission = _net_conf(word)

    return crf_decode, word, emission
