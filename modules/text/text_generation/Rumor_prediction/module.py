# coding:utf-8
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
import argparse
import ast
import os
import math
import six
import time
from pathlib import Path

from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import runnable, serving, moduleinfo
from paddlehub.io.parser import txt_parser
from paddlehub.compat.module.nlp_module import DataFormatError
import numpy as np
import paddle
import paddlehub as hub

@moduleinfo(
    name="Rumor_prediction",
    version="1.0.0",
    type="nlp/semantic_model",
    summary=
    "Is the input text prediction a rumor",
    author="彭兆帅，郑博培",
    author_email="1084667371@qq.com，2733821739@qq.com")
class Rumorprediction(hub.Module):
    def _initialize(self):
        """
        Initialize with the necessary elements
        """
        # 加载模型路径
        self.default_pretrained_model_path = os.path.join(self.directory, "infer_model")
    
    def Rumor(self, texts, use_gpu=False):
        """
        Get the input and program of the infer model

        Args:
             image (list(numpy.ndarray)): images data, shape of each is [H, W, C], the color space is BGR.
             use_gpu(bool): Weather to use gpu
        """
        # 获取数据
        def get_data(sentence):
            # 读取数据字典
            with open(self.directory + '/dict.txt', 'r', encoding='utf-8') as f_data:
                dict_txt = eval(f_data.readlines()[0])
            dict_txt = dict(dict_txt)
            # 把字符串数据转换成列表数据
            keys = dict_txt.keys()
            data = []
            for s in sentence:
                # 判断是否存在未知字符
                if not s in keys:
                    s = '<unk>'
                data.append(int(dict_txt[s]))
            return data
        data = []
        for text in texts:
            text = get_data(text)
            data.append(text)
        base_shape = [[len(c) for c in data]]
        paddle.enable_static()
        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        [infer_program, feeded_var_names, target_var] = paddle.fluid.io.load_inference_model(dirname=self.default_pretrained_model_path, executor=exe)
        # 生成预测数据
        tensor_words = paddle.fluid.create_lod_tensor(data, base_shape, place)
        # 执行预测
        result = exe.run(program=infer_program,
                        feed={feeded_var_names[0]: tensor_words},
                        fetch_list=target_var)
        # 分类名称
        names = [ '谣言', '非谣言']


        results = []

        # 获取结果概率最大的label
        for i in range(len(data)):
            content = texts[i]
            lab = np.argsort(result)[0][i][-1]

            alltext = {
                'content': content,
                'prediction': names[lab],
                'probability': result[0][i][lab]
            }
            alltext = [alltext]
            results = results + alltext
            # results = results.append(alltext)
            # print(alltext)
            # results = results.append(alltext)
            
        return results

    
    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU for prediction")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_text',
            type=str,
            default=None,
            help="input_text is str")
    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, optional.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)
        input_text = [args.input_text]
        results = self.Rumor(
            texts=input_text, use_gpu=args.use_gpu)

        return results
