# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.common.paddle_helper import add_vars_prefix
from paddlehub.module.module import moduleinfo


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.split("\t")
            vocab[parts[0]] = int(parts[1])

    return vocab


@moduleinfo(
    name="tencent_ailab_chinese_embedding",
    version="1.0.0",
    summary=
    "Tencent AI Lab Embedding Corpus for Chinese Words and Phrases and the vocab size is 8,824,331. For more information, please refer to https://ai.tencent.com/ailab/nlp/zh/embedding.html",
    author="",
    author_email="",
    type="nlp/semantic_model")
class TencentAILabChineseEmbedding(hub.Module):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets",
                                                  "model")
        self.vocab_path = os.path.join(self.directory, "assets", "vocab.txt")
        self.vocab = load_vocab(self.vocab_path)

    def context(self, trainable=False, max_seq_len=128, num_slots=1):
        """
        Get the input ,output and program of the pretrained tencent_ailab_chinese_embedding

        Args:
             trainable(bool): whether fine-tune the pretrained parameters of simnet_bow or not
             num_slots(int): It's number of slots inputted to the model, selectted as following options:

                 - 1(default): There's only one data to be feeded in the model, e.g. the module is used for sentence classification task.
                 - 2: There are two data to be feeded in the model, e.g. the module is used for text matching task (point-wise).
                 - 3: There are three data to be feeded in the model, e.g. the module is used for text matching task (pair-wise).

        Returns:
             inputs(dict): the input variables of tencent_ailab_chinese_embedding (words)
             outputs(dict): the output variables of input words (word embeddings)
             main_program(Program): the main_program of tencent_ailab_chinese_embedding with pretrained prameters
        """
        assert num_slots >= 1 and num_slots <= 3, "num_slots must be 1, 2, or 3, but the input is %d" % num_slots
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard():
                w_param_attrs = fluid.ParamAttr(
                    name="embedding_0.w_0",
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02),
                    trainable=trainable)

                text_1 = fluid.data(
                    name='text',
                    shape=[-1, max_seq_len],
                    dtype='int64',
                    lod_level=0)
                emb_1 = fluid.embedding(
                    input=text_1,
                    size=[len(self.vocab), 200],
                    is_sparse=True,
                    padding_idx=len(self.vocab) - 1,
                    dtype='float32',
                    param_attr=w_param_attrs)
                emb_1_name = emb_1.name
                data_list = [text_1]
                emb_name_list = [emb_1_name]

                if num_slots > 1:
                    text_2 = fluid.data(
                        name='text_2',
                        shape=[-1, max_seq_len],
                        dtype='int64',
                        lod_level=0)
                    emb_2 = fluid.embedding(
                        input=text_2,
                        size=[len(self.vocab), 200],
                        is_sparse=True,
                        padding_idx=len(self.vocab) - 1,
                        dtype='float32',
                        param_attr=w_param_attrs)
                    emb_2_name = emb_2.name
                    data_list.append(text_2)
                    emb_name_list.append(emb_2_name)

                if num_slots > 2:
                    text_3 = fluid.data(
                        name='text_3',
                        shape=[-1, max_seq_len],
                        dtype='int64',
                        lod_level=0)
                    emb_3 = fluid.embedding(
                        input=text_3,
                        size=[len(self.vocab), 200],
                        is_sparse=True,
                        padding_idx=len(self.vocab) - 1,
                        dtype='float32',
                        param_attr=w_param_attrs)
                    emb_3_name = emb_3.name
                    data_list.append(text_3)
                    emb_name_list.append(emb_3_name)

                variable_names = filter(
                    lambda v: v not in ['text', 'text_2', 'text_3'],
                    list(main_program.global_block().vars.keys()))

                prefix_name = "@HUB_{}@".format(self.name)
                add_vars_prefix(
                    program=main_program,
                    prefix=prefix_name,
                    vars=variable_names)
                for param in main_program.global_block().iter_parameters():
                    param.trainable = trainable

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)

                # load the pretrained model
                def if_exist(var):
                    return os.path.exists(
                        os.path.join(self.pretrained_model_path, var.name))

                fluid.io.load_vars(
                    exe, self.pretrained_model_path, predicate=if_exist)

                inputs = {}
                outputs = {}
                for index, data in enumerate(data_list):
                    if index == 0:
                        inputs['text'] = data
                        outputs['emb'] = main_program.global_block().vars[
                            prefix_name + emb_name_list[0]]
                    else:
                        inputs['text_%s' % (index + 1)] = data
                        outputs['emb_%s' %
                                (index + 1)] = main_program.global_block().vars[
                                    prefix_name + emb_name_list[index]]

                return inputs, outputs, main_program

    def get_vocab_path(self):
        return self.vocab_path


if __name__ == "__main__":
    w2v = TencentAILabChineseEmbedding()
    inputs, outputs, program = w2v.context(num_slots=3)
    print(inputs)
    print(outputs)
    print(w2v.get_vocab_path())
