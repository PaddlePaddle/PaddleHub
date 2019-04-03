import paddle
import paddle.fluid as fluid
import paddlehub as hub
import numpy as np
import os
import io
from paddlehub import BaseProcessor


class Processor(BaseProcessor):
    def __init__(self, module):
        self.module = module
        assets_path = self.module.helper.assets_path()
        word_dict_path = os.path.join(assets_path, "word.dic")
        label_dict_path = os.path.join(assets_path, "tag.dic")
        word_rep_dict_path = os.path.join(assets_path, "q2b.dic")

        self.id2word_dict = self.load_dict(word_dict_path)
        self.word2id_dict = self.load_reverse_dict(word_dict_path)
        self.id2label_dict = self.load_dict(label_dict_path)
        self.label2id_dict = self.load_reverse_dict(label_dict_path)
        self.q2b_dict = self.load_dict(word_rep_dict_path)

    def load_dict(self, dict_path):
        result_dict = {}
        for line in io.open(dict_path, "r", encoding='utf8'):
            terms = line.strip("\n").split("\t")
            if len(terms) != 2:
                continue
            result_dict[terms[0]] = terms[1]
        return result_dict

    def load_reverse_dict(self, dict_path):
        result_dict = {}
        for line in io.open(dict_path, "r", encoding='utf8'):
            terms = line.strip("\n").split("\t")
            if len(terms) != 2:
                continue
            result_dict[terms[1]] = terms[0]
        return result_dict

    def preprocess(self, sign_name, data_dict):
        result = {'text': []}
        for sentence in data_dict['text']:
            result_i = {}
            result_i['origin'] = sentence
            line = sentence.strip()
            word_idx = []
            for word in line:
                if ord(word) < 0x20:
                    word = ' '
                if word in self.q2b_dict:
                    word = self.q2b_dict[word]
                if word in self.word2id_dict:
                    word_idx.append(int(self.word2id_dict[word]))
                else:
                    word_idx.append(int(self.word2id_dict["OOV"]))

            result_i['attach'] = line
            result_i['processed'] = [x for x in word_idx]
            result['text'].append(result_i)
        return result

    def postprocess(self, sign_name, data_out, data_info, **kwargs):
        if sign_name == "lexical_analysis":
            result = []
            crf_decode = data_out[0]
            lod_info = (crf_decode.lod())[0]
            np_data = np.array(crf_decode)
            for index in range(len(lod_info) - 1):
                seg_result = {"word": [], "tag": []}
                word_index = 0
                outstr = ""
                offset = 0
                cur_full_word = ""
                cur_full_tag = ""
                words = data_info['text'][index]['attach']
                for tag_index in range(lod_info[index], lod_info[index + 1]):
                    cur_word = words[word_index]
                    cur_tag = self.id2label_dict[str(np_data[tag_index][0])]
                    if cur_tag.endswith("-B") or cur_tag.endswith("O"):
                        if len(cur_full_word) != 0:
                            seg_result['word'].append(cur_full_word)
                            seg_result['tag'].append(cur_full_tag)
                        cur_full_word = cur_word
                        cur_full_tag = self.get_real_tag(cur_tag)
                    else:
                        cur_full_word += cur_word
                    word_index += 1
                seg_result['word'].append(cur_full_word)
                seg_result['tag'].append(cur_full_tag)
                result.append(seg_result)
            return result

    def get_real_tag(self, origin_tag):
        if origin_tag == "O":
            return "O"
        return origin_tag[0:len(origin_tag) - 2]

    def data_format(self, sign_name):
        if sign_name == "lexical_analysis":
            return {
                "text": {
                    "type": hub.DataType.TEXT,
                    "feed_key": self.module.signatures[sign_name].inputs[0].name
                }
            }
        return None
