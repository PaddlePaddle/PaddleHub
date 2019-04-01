import numpy as np
import paddle
import paddle.fluid as fluid
import reader
import paddle_hub as hub
import processor
import os
from network import lex_net


def create_module():
    word_dict_path = "resources/word.dic"
    label_dict_path = "resources/tag.dic"
    word_rep_dict_path = "resources/q2b.dic"
    pretrained_model = "resources/model"

    word2id_dict = reader.load_reverse_dict(word_dict_path)
    label2id_dict = reader.load_reverse_dict(label_dict_path)
    word_rep_dict = reader.load_dict(word_rep_dict_path)
    word_dict_len = max(map(int, word2id_dict.values())) + 1
    label_dict_len = max(map(int, label2id_dict.values())) + 1

    avg_cost, crf_decode, word, target = lex_net(word_dict_len, label_dict_len)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load the lac pretrained model
    def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))

    fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    # assets
    assets = [word_dict_path, label_dict_path, word_rep_dict_path]

    # create a module and save as hub_module_lac
    sign = hub.create_signature(
        name="lexical_analysis",
        inputs=[word],
        outputs=[crf_decode],
        for_predict=True)
    hub.create_module(
        sign_arr=[sign],
        module_dir="hub_module_lac",
        exe=exe,
        module_info="resources/module_info.yml",
        processor=processor.Processor,
        assets=assets)


if __name__ == "__main__":
    create_module()
