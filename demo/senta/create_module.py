import io
import paddle.fluid as fluid
import processor
import numpy as np
import nets
import paddlehub as hub


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            vocab[parts[0]] = int(parts[1])
    vocab["<unk>"] = len(vocab)
    return vocab


def create_module():
    network = nets.bilstm_net
    # word seq data
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    # label data
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
    word_dict_path = "./resources/train.vocab"
    word_dict = load_vocab(word_dict_path)
    cost, acc, pred = network(data, label, len(word_dict) + 1)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    model_path = "./resources/senta_model"
    fluid.io.load_inference_model(model_path, exe)

    # assets
    assets = [word_dict_path]

    # create a module
    sign = hub.create_signature(
        name="sentiment_classify",
        inputs=[data],
        outputs=[pred],
        for_predict=True)
    hub.create_module(
        sign_arr=[sign],
        module_dir="senta.hub_module",
        exe=exe,
        module_info="resources/module_info.yml",
        processor=processor.Processor,
        assets=assets)


if __name__ == "__main__":
    create_module()
