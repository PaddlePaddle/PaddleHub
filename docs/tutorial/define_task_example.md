# 如何修改Task中的模型网络

在应用中，用户需要更换迁移网络结构以调整模型在数据集上的性能。根据[如何自定义Task](./how_to_define_task.md)，本教程展示如何修改Task中的默认网络。
以序列标注任务为例，本教程展示如何修改默认网络结构。SequenceLabelTask提供了两种网络选择，一种是FC网络，一种是FC+CRF网络。

此时如果想在这基础之上，添加LSTM网络，组成BiLSTM+CRF的一种序列标注任务常用网络结构。
此时，需要定义一个Task，继承自SequenceLabelTask，并改写其中build_net()方法。


下方代码示例写了一个BiLSTM+CRF的网络。代码如下：

```python
class SequenceLabelTask_BiLSTMCRF(SequenceLabelTask):
    def _build_net(self):
        """
        自定义序列标注迁移网络结构BiLSTM+CRF
        """
        self.seq_len = fluid.layers.data(
            name="seq_len", shape=[1], dtype='int64', lod_level=0)

        if version_compare(paddle.__version__, "1.6"):
            self.seq_len_used = fluid.layers.squeeze(self.seq_len, axes=[1])
        else:
            self.seq_len_used = self.seq_len

        if self.add_crf:
            # 迁移网络为BiLSTM+CRF

            # 去padding
            unpad_feature = fluid.layers.sequence_unpad(
                self.feature, length=self.seq_len_used)

            # bilstm层
            hid_dim = 128
            fc0 = fluid.layers.fc(input=unpad_feature, size=hid_dim * 4)
            rfc0 = fluid.layers.fc(input=unpad_feature, size=hid_dim * 4)
            lstm_h, c = fluid.layers.dynamic_lstm(
                input=fc0, size=hid_dim * 4, is_reverse=False)
            rlstm_h, c = fluid.layers.dynamic_lstm(
                input=rfc0, size=hid_dim * 4, is_reverse=True)
            # 拼接lstm
            lstm_concat = fluid.layers.concat(input=[lstm_h, rlstm_h], axis=1)

            self.emission = fluid.layers.fc(
                size=self.num_classes,
                input=lstm_concat,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(low=-0.1, high=0.1),
                    regularizer=fluid.regularizer.L2DecayRegularizer(
                        regularization_coeff=1e-4)))
            size = self.emission.shape[1]
            fluid.layers.create_parameter(
                shape=[size + 2, size], dtype=self.emission.dtype, name='crfw')
            # CRF层
            self.ret_infers = fluid.layers.crf_decoding(
                input=self.emission, param_attr=fluid.ParamAttr(name='crfw'))
            ret_infers = fluid.layers.assign(self.ret_infers)
            # 返回预测值，list类型
            return [ret_infers]
        else:
            # 迁移网络为FC
            self.logits = fluid.layers.fc(
                input=self.feature,
                size=self.num_classes,
                num_flatten_dims=2,
                param_attr=fluid.ParamAttr(
                    name="cls_seq_label_out_w",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=fluid.ParamAttr(
                    name="cls_seq_label_out_b",
                    initializer=fluid.initializer.Constant(0.)))

            self.ret_infers = fluid.layers.reshape(
                x=fluid.layers.argmax(self.logits, axis=2), shape=[-1, 1])

            logits = self.logits
            logits = fluid.layers.flatten(logits, axis=2)
            logits = fluid.layers.softmax(logits)
            self.num_labels = logits.shape[1]
            # 返回预测值，list类型
            return [logits]
```

以上代码通过继承PaddleHub已经内置的Task，改写其中_build_net方法即可实现自定义迁移网络结构。
