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
import os

from collections import defaultdict
from paddle_hub.downloader import download_and_uncompress
from paddle_hub import module_desc_pb2

__all__ = ["Module", "ModuleConfig", "ModuleUtils"]
DICT_NAME = "dict.txt"
ASSETS_NAME = "assets"


def mkdir(path):
    """ the same as the shell command mkdir -p "
    """
    if not os.path.exists(path):
        os.makedirs(path)


class Module(object):
    def __init__(self, module_url=None, module_dir=None):
        if module_url == None and module_dir == None:
            raise Exception("Module:module_url and module_dir are None!")

        self.module_dir = ""
        self.module_name = ""
        # donwload module
        if module_url is not None and module_url.startswith("http"):
            # if it's remote url link, then download and uncompress it
            self.module_name, self.module_dir = download_and_uncompress(
                module_url)
            #TODO(ZeyuChen): check url link is valid url
        elif module_dir is not None:
            # otherwise it's local path, no need to deal with it
            self.module_dir = module_dir
            # use the path name as module name by default
            self.module_name = module_dir.split("/")[-1]
            #TODO(ZeyuChen) add more check about loading module from local path

        # load paddle inference model
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(fluid.CPUPlace())
        [self.inference_program, self.feed_target_names,
         self.fetch_targets] = fluid.io.load_inference_model(
             dirname=self.module_dir, executor=self.exe)

        print("inference_program")
        print(self.inference_program)
        print("feed_target_names")
        print(self.feed_target_names)
        print("fetch_targets")
        print(self.fetch_targets)

        self.config = ModuleConfig(self.module_dir)
        self.config.load()
        # load assets
        # self.dict = defaultdict(int)
        # self.dict.setdefault(0)
        # self._load_assets(module_dir)

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

    def get_feed_var(self, key, signature="default"):
        for var in self.inference_program.list_vars():
            if var.name == self.config.feed_var_name(key, signature):
                return var

        raise Exception("Can't find input var {}".format(key))

    def get_fetch_var(self, key, signature="default"):
        for var in self.inference_program.list_vars():
            if var.name == self.config.fetch_var_name(key, signature):
                return var

        raise Exception("Can't find output var {}".format(key))

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
        word_dict = self.config.get_dict()
        return list(map(lambda x: word_dict[x], inputs))


class ModuleConfig(object):
    def __init__(self, module_dir, module_name=None):
        # generate model desc protobuf
        self.module_dir = module_dir
        self.desc = module_desc_pb2.ModuleDesc()
        if module_name == None:
            module_name = module_dir.split("/")[-1]
        self.desc.name = module_name
        print("desc.name=", self.desc.name)
        self.desc.contain_assets = True
        print("desc.signature=", self.desc.contain_assets)

        # init dict
        self.dict = defaultdict(int)
        self.dict.setdefault(0)

    def load(self):
        """load module config from module dir
        """
        #TODO(ZeyuChen): check module_desc.pb exsitance
        pb_path = os.path.join(self.module_dir, "module_desc.pb")
        with open(pb_path, "rb") as fi:
            self.desc.ParseFromString(fi.read())

        if self.desc.contain_assets:
            # load assets
            assets_dir = os.path.join(self.module_dir, ASSETS_NAME)
            dict_path = os.path.join(assets_dir, DICT_NAME)
            word_id = 0

            with open(dict_path) as fi:
                words = fi.readlines()
                #TODO(ZeyuChen) check whether word id is duplicated and valid
                for line in fi:
                    w, w_id = line.split()
                    self.dict[w] = int(w_id)

    def dump(self):
        """
        save module_desc.proto first
        """
        pb_path = os.path.join(self.module_dir, "module_desc.pb")
        with open(pb_path, "wb") as fo:
            fo.write(self.desc.SerializeToString())

        # save assets/dictionary
        assets_dir = os.path.join(self.module_dir, ASSETS_NAME)
        mkdir(assets_dir)
        with open(os.path.join(assets_dir, DICT_NAME), "w") as fo:
            for w in self.dict:
                w_id = self.dict[w]
                fo.write("{}\t{}\n".format(w, w_id))

    def save_dict(self, word_dict, dict_name=DICT_NAME):
        """ Save dictionary for NLP module
        """
        for w in word_dict:
            self.dict[w] = word_dict[w]
        # mkdir(self.module_dir)
        # with open(os.path.join(self.module_dir, DICT_NAME), "w") as fo:
        #     for w in word_dict:
        #         self.dict[w] = word_dict[w]

    def get_dict(self):
        return self.dict

    def register_feed_signature(self, feed_desc, sign_name="default"):
        #TODO(ZeyuChen) check fetch_desc key is valid and no duplicated
        for k in feed_desc:
            feed = self.desc.sign2var[sign_name].feed_desc.add()
            feed.key = k
            feed.var_name = feed_desc[k]

    def register_fetch_signature(self, fetch_desc, sign_name="default"):
        #TODO(ZeyuChen) check fetch_desc key is valid and no duplicated
        for k in fetch_desc:
            fetch = self.desc.sign2var[sign_name].fetch_desc.add()
            fetch.key = k
            fetch.var_name = fetch_desc[k]

    def feed_var_name(self, key, sign_name="default"):
        for desc in self.desc.sign2var[sign_name].feed_desc:
            if desc.key == key:
                return desc.var_name
        raise Exception("feed variable {} not found".format(key))

    def fetch_var_name(self, key, sign_name="default"):
        for desc in self.desc.sign2var[sign_name].fetch_desc:
            if desc.key == key:
                return desc.var_name
        raise Exception("fetch variable {} not found".format(key))


class ModuleUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def remove_feed_fetch_op(program):
        """ remove feed and fetch operator and variable for fine-tuning
        """
        print("remove feed fetch op")
        block = program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(block.ops):
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            block._remove_op(index)

        block._remove_var("feed")
        block._remove_var("fetch")

        program.desc.flush()
        # print("********************************")
        # print(program)
        # print("********************************")


if __name__ == "__main__":
    url = "http://paddlehub.cdn.bcebos.com/word2vec/word2vec-dim16-simple-example-2.tar.gz"
    m = Module(module_url=url)
    inputs = [["it", "is", "new"], ["hello", "world"]]
    #tensor = m._process_input(inputs)
    #print(tensor)
    result = m(inputs)
    print(result)
