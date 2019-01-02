#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import numpy as np
import tempfile
import utils
import os

from collections import defaultdict
from downloader import download_and_uncompress

__all__ = ["Module", "ModuleDesc"]
DICT_NAME = "dict.txt"
ASSETS_PATH = "assets"


def mkdir(path):
    """ the same as the shell command mkdir -p "
    """
    if not os.path.exists(path):
        os.makedirs(path)


class Module(object):
    def __init__(self, module_url):
        # donwload module
        if module_url.startswith("http"):  # if it's remote url links
            # if it's remote url link, then download and uncompress it
            module_dir = download_and_uncompress(module_url)
        else:
            # otherwise it's local path, no need to deal with it
            module_dir = module_url

        # load paddle inference model
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(fluid.CPUPlace())
        [self.inference_program, self.feed_target_names,
         self.fetch_targets] = fluid.io.load_inference_model(
             dirname=module_dir, executor=self.exe)

        print("inference_program")
        print(self.inference_program)
        print("feed_target_names")
        print(self.feed_target_names)
        print("fetch_targets")
        print(self.fetch_targets)

        # load assets
        self.dict = defaultdict(int)
        self.dict.setdefault(0)
        self._load_assets(module_dir)

    #TODO(ZeyuChen): Need add register more signature to execute different
    # implmentation
    def __call__(self, inputs=None, signature=None):
        """ Call default signature and return results
        """
        # TODO(ZeyuChen): add proto spec to check which task we need to run
        # if it's NLP word embedding task, then do words preprocessing
        # if it's image classification or image feature task do the other works

        # if it's
        word_ids_lod_tensor = self._process_input(inputs)
        np_words_id = np.array(word_ids_lod_tensor)
        print("word_ids_lod_tensor\n", np_words_id)

        results = self.exe.run(
            self.inference_program,
            feed={self.feed_target_names[0]: word_ids_lod_tensor},
            fetch_list=self.fetch_targets,
            return_numpy=False)  # return_numpy=Flase is important

        print("module fetch_target_names", self.feed_target_names)
        print("module fetch_targets", self.fetch_targets)
        np_result = np.array(results[0])

        return np_result

    def get_vars(self):
        return self.inference_program.list_vars()

    def get_input_vars(self):
        for var in self.inference_program.list_vars():
            print(var)
            if var.name == "words":
                return var
        # return self.fetch_targets

    def get_module_output(self):
        for var in self.inference_program.list_vars():
            print(var)
            # NOTE: just hack for load Senta's
            if var.name == "embedding_0.tmp_0":
                return var

    def get_inference_program(self):
        return self.inference_program

    # for text sequence input, transform to lod tensor as paddle graph's input
    def _process_input(self, inputs):
        # words id mapping and dealing with oov
        # transform to lod tensor
        seq = []
        for s in inputs:
            seq.append(self._word_id_mapping(s))

        lod_tensor = self.seq2lod_tensor(seq)

        return lod_tensor

    def seq2lod_tensor(self, seq_inputs, place=fluid.CPUPlace()):
        """ sequence to lod tensor, need to determine which space"""
        lod = []
        lod.append([])
        for s in seq_inputs:
            # generate lod
            lod[0].append(len(s))

        # print("seq", seq_inputs)
        # print("lod", lod)

        lod_tensor = fluid.create_lod_tensor(seq_inputs, lod, place)

        return lod_tensor

    def _word_id_mapping(self, inputs):
        return list(map(lambda x: self.dict[x], inputs))

    # load assets folder
    def _load_assets(self, module_dir):
        assets_dir = os.path.join(module_dir, ASSETS_PATH)
        tokens_path = os.path.join(assets_dir, DICT_NAME)
        word_id = 0

        with open(tokens_path) as fi:
            words = fi.readlines()
            #TODO(ZeyuChen) check whether word id is duplicated and valid
            for line in fi:
                w, w_id = line.split()
                self.dict[w] = int(w_id)

            # words = map(str.strip, words)
            # for w in words:
            #     self.dict[w] = word_id
            #     word_id += 1
            #     print(w, word_id)

    def add_module_feed_list(self, feed_list):
        self.feed_list = feed_list

    def add_module_output_list(self, output_list):
        self.output_list = output_list


class ModuleDesc(object):
    def __init__(self):
        pass

    @staticmethod
    def save_dict(path, word_dict, dict_name):
        """ Save dictionary for NLP module
        """
        mkdir(path)
        with open(os.path.join(path, dict_name), "w") as fo:
            print("tokens.txt path", os.path.join(path, DICT_NAME))
            for w in word_dict:
                w_id = word_dict[w]
                fo.write("{}\t{}\n".format(w, w_id))

    @staticmethod
    def save_module_dict(module_path, word_dict, dict_name=DICT_NAME):
        """ Save dictionary for NLP module
        """
        assets_path = os.path.join(module_path, ASSETS_PATH)
        print("save_module_dict", assets_path)
        ModuleDesc.save_dict(assets_path, word_dict, dict_name)
        pass


if __name__ == "__main__":
    module_link = "http://paddlehub.cdn.bcebos.com/word2vec/w2v_saved_inference_module.tar.gz"
    m = Module(module_link)
    inputs = [["it", "is", "new"], ["hello", "world"]]
    #tensor = m._process_input(inputs)
    #print(tensor)
    result = m(inputs)
    print(result)
