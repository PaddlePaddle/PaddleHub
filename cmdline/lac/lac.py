#coding: utf-8

from __future__ import print_function
import numpy as np
import paddle.fluid as fluid
import argparse
import time
import sys
import io

def parse_args():
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='The size of a batch. (default: %(default)d)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./conf/model',
        help='A path to the model. (default: %(default)s)'
    )
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='./data/test_data',
        help='A directory with test data files. (default: %(default)s)'
    )
    parser.add_argument(
        "--word_dict_path",
        type=str,
        default="./conf/word.dic",
        help="The path of the word dictionary. (default: %(default)s)"
    )
    parser.add_argument(
        "--label_dict_path",
        type=str,
        default="./conf/tag.dic",
        help="The path of the label dictionary. (default: %(default)s)"
    )
    parser.add_argument(
        "--word_rep_dict_path",
        type=str,
        default="./conf/q2b.dic",
        help="The path of the word replacement Dictionary. (default: %(default)s)"
    )
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def get_real_tag(origin_tag):
    if origin_tag == "O":
        return "O"
    return origin_tag[0:len(origin_tag) - 2]

# Object oriented encapsulate paddle model inference
class LACModel(object):
    def __init__(self, args):
        self.place = fluid.CPUPlace() # LAC use CPU place as default
        self.exe = fluid.Executor(self.place)
        # initialize dictionary
        self.id2word_dict = self.load_dict(args.word_dict_path)
        self.word2id_dict = self.load_reverse_dict(args.word_dict_path) 

        self.id2label_dict = self.load_dict(args.label_dict_path)
        self.label2id_dict = self.load_reverse_dict(args.label_dict_path)
        self.q2b_dict = self.load_dict(args.word_rep_dict_path)

        self.inference_program, self.feed_target_names, self.fetch_targets = fluid.io.load_inference_model(args.model_path, self.exe)

    def download_module(self):
        pass

    def preprocess(self, sentence):
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

        word_idx_list = [[x for x in word_idx]]
        print(word_idx_list)
        word_idx_lod = self.__to_lodtensor(word_idx_list, self.place)

        word_list = [line]
        print(word_list)
        return word_idx_lod, word_list


    def segment(self, sentence):
        sentence = sentence.strip()
        full_out_str = ""
        word_idx_lod, word_list = self.preprocess(sentence)
        (crf_decode, ) = self.exe.run(self.inference_program,
                             feed={"word":word_idx_lod},
                             fetch_list=self.fetch_targets,
                             return_numpy=False)

        lod_info = (crf_decode.lod())[0]
        print(lod_info)
        np_data = np.array(crf_decode)
        print(np_data)
        #assert len(data) == len(lod_info) - 1
        for sen_index in range(len(word_list)):
            word_index = 0
            outstr = ""
            cur_full_word = ""
            cur_full_tag = ""
            words = word_list[sen_index]
            for tag_index in range(lod_info[sen_index],
                                    lod_info[sen_index + 1]):
                cur_word = words[word_index]
                cur_tag = self.id2label_dict[str(np_data[tag_index][0])]
                if cur_tag.endswith("-B") or cur_tag.endswith("O"):
                    if len(cur_full_word) != 0:
                        outstr += cur_full_word + u"/" + cur_full_tag + u" "
                    cur_full_word = cur_word
                    cur_full_tag = get_real_tag(cur_tag)
                else:
                    cur_full_word += cur_word
                word_index += 1
            outstr += cur_full_word + u"/" + cur_full_tag + u" "    
            outstr = outstr.strip()
            full_out_str += outstr + u"\n"
        print(full_out_str.strip(), file=sys.stdout)

    def ner(self, sentence):
        pass

    def postag(self, sentence):
        pass

    def __to_lodtensor(self, data, place):
        seq_lens = [len(seq) for seq in data]
        cur_len = 0
        lod = [cur_len]
        for l in seq_lens:
            cur_len += l
            lod.append(cur_len)
        flattened_data = np.concatenate(data, axis=0).astype("int64")
        flattened_data = flattened_data.reshape([len(flattened_data), 1])
        res = fluid.LoDTensor()
        res.set(flattened_data, place)
        res.set_lod([lod])
        return res

    def load_dict(self, dict_path):
        """
        Load a dict. The first column is the key and the second column is the value.
        """
        result_dict = {}
        for line in io.open(dict_path, "r", encoding='utf8'):
            terms = line.strip("\n").split("\t")
            if len(terms) != 2:
                continue
            result_dict[terms[0]] = terms[1]
        return result_dict

    def load_reverse_dict(self, dict_path):
        """
        Load a dict. The first column is the value and the second column is the key.
        """
        result_dict = {}
        for line in io.open(dict_path, "r", encoding='utf8'):
            terms = line.strip("\n").split("\t")
            if len(terms) != 2:
                continue
            result_dict[terms[1]] = terms[0]
        return result_dict


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    lac = LACModel(args)
    lac.segment("我是一个中国人")
