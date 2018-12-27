from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
import paddle_hub as hub
import tempfile
import os


class Module(object):
    def __init__(self, module_url):
        module_dir = download_and_extract(module_url)

    def __init__(self, module_name, module_dir=None):
        if module_dir is None:
            self.module_dir = tempfile.gettempdir()
        else:
            self.module_dir = module_dir

        self.module_name = module_name
        self.module_dir = os.path.join(self.module_dir, self.module_name)
        print("create module dir folder at {}".format(self.module_dir))
        self._mkdir(self.module_dir)

        self.feed_list = []
        self.output_list = []
        pass

    def save_dict(self, word_dict):
        with open(os.path.join(self.module_dir, "tokens.txt"), "w") as fo:
            #map(str, word_dict)
            dict_str = "\n".join(word_dict)
            fo.write(dict_str)

    def add_module_feed_list(self, feed_list):
        self.feed_list = feed_list

    def add_module_output_list(self, output_list):
        self.output_list = output_list

    def _mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_inference_module(feed_var_names, target_vars, executor):
        pass


class ModuleImpl(object):
    def get_signature_name():
        pass


class ModuleDesc(object):
    def __init__(self, input_list, output_list):
        self.input_list = input_sig
        self.output_list = output_list
        pass

    def add_signature(input, output):
        pass
