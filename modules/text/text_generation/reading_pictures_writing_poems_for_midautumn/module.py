import argparse
import ast
import os
import math
import six

from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import runnable, serving, moduleinfo
from paddlehub.io.parser import txt_parser
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from translate import Translator
import reading_pictures_writing_poems_for_midautumn.MidAutumnDetection.module as MidAutumnDetection
import reading_pictures_writing_poems_for_midautumn.MidAutumnPoetry.module as MidAutumnPoetry


@moduleinfo(
    name="reading_pictures_writing_poems_for_midautumn",
    version="1.0.0",
    summary="Reading Pictures And Writing Poems For MidAutumn",
    author="郑博培，彭兆帅",
    author_email="2733821739@qq.com，1084667371@qq.com",
    type="nlp/text_generation")
class ReadingPicturesWritingPoems(hub.Module):
    def _initialize(self):
        """
        Initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets", "infer_model")
        self.module_image = MidAutumnDetection.MODULE(
            directory="reading_pictures_writing_poems_for_midautumn/MidAutumnDetection")  # 调用目标检测的模型
        self.module_similar = MidAutumnPoetry.ErnieGen(
            directory='reading_pictures_writing_poems_for_midautumn/MidAutumnPoetry')  # 调用根据关键词生成古诗上阕的模型
        self.module_poem = hub.Module(name="ernie_gen_poetry")  # 调用古诗生成的模型

    def WritingPoem(self, images, use_gpu=False):
        # 目标检测，输入图片，输入得分最高的标签
        results_image = self.module_image.predict(images=images)
        best = {'score': 0, 'category': 'none'}
        for item in results_image:
            for items in item:
                if (items['score'] > best['score']):
                    best['score'], best['category'] = items['score'], items['category']
        if best['category'] == 'MoonCake':
            objects = ['月饼']
        elif best['category'] == 'moon':
            objects = ['月亮']
        elif best['category'] == 'lantern':
            objects = ['灯笼']
        elif best['category'] == 'rabbit':
            objects = ['兔子']
        else:
            objects = ['中秋节']
        # 根据关键词生成古诗上阕
        FirstPoetrys = self.module_similar.generate(texts=objects, use_gpu=True, beam_width=5)
        FirstPoetry = [FirstPoetrys[0][0]]
        # 调用古诗生成模型，使用上阕生成下阕
        SecondPoetry = self.module_poem.generate(texts=FirstPoetry, use_gpu=True, beam_width=5)
        Poetrys = []
        Poetrys.append(FirstPoetry[0])
        Poetrys.append(SecondPoetry[0][0])
        results = [{'images': images, 'Poetrys': "{}".format(Poetrys[0] + Poetrys[1])}]

        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except RuntimeError:
            self.parser.print_help()
            return None

        results = self.WritingPoem(input_data)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU for prediction")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument('--input_image', type=str, default=None, help="Pictures to write poetry")

    def check_input_data(self, args):
        input_data = []
        if args.input_image:
            if not os.path.exists(args.input_image):
                raise RuntimeError("File %s is not exist." % args.input_image)
            else:
                input_data = args.input_image

        if input_data == []:
            raise RuntimeError("The input data is inconsistent with expectations.")

        return input_data
