import os
import io

import paddle
import paddle.fluid as fluid
import numpy as np

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


def get_predict_label(pos_prob):
    neg_prob = 1 - pos_prob
    # threshold should be (1, 0.5)
    neu_threshold = 0.55
    if neg_prob > neu_threshold:
        label, key = 0, "负面"
    elif pos_prob > neu_threshold:
        label, key = 2, "正面"
    else:
        label, key = 1, "中性"
    return label, key


class Processor(hub.BaseProcessor):
    def __init__(self, module):
        self.module = module
        assets_path = self.module.helper.assets_path()
        word_dict_path = os.path.join(assets_path, "train.vocab")
        self.word_dict = load_vocab(word_dict_path)
        path = hub.default_module_manager.search_module("lac")
        if path:
            self.lac = hub.Module(module_dir=path)
        else:
            result, _, path = hub.default_module_manager.install_module("lac")
            assert path, "can't found necessary module lac"
            self.lac = hub.Module(module_dir=path)

    def preprocess(self, sign_name, data_dict):
        result = {'text': []}
        processed = self.lac.lexical_analysis(data=data_dict)
        unk_id = len(self.word_dict)
        for index, data in enumerate(processed):
            result_i = {'processed': []}
            result_i['origin'] = data_dict['text'][index]
            for word in data['word']:
                if word in self.word_dict:
                    _index = self.word_dict[word]
                else:
                    _index = unk_id
                result_i['processed'].append(_index)
            result['text'].append(result_i)
        return result

    def postprocess(self, sign_name, data_out, data_info, **kwargs):
        if sign_name == "sentiment_classify":
            result = []
            pred = fluid.executor.as_numpy(data_out)
            for index in range(len(data_info['text'])):
                result_i = {}
                result_i['text'] = data_info['text'][index]['origin']
                label, key = get_predict_label(pred[0][index, 1])
                result_i['sentiment_label'] = label
                result_i['sentiment_key'] = key
                result.append(result_i)
            return result

    def data_format(self, sign_name):
        if sign_name == "sentiment_classify":
            return {
                "text": {
                    "type": hub.DataType.TEXT,
                    "feed_key": self.module.signatures[sign_name].inputs[0].name
                }
            }
        return None
