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

@moduleinfo(
    name="reading_pictures_writing_poems",
    version="1.0.0",
    summary="Just for test",
    author="Mr.郑先生_",
    author_email="2733821739@qq.com",
    type="nlp/text_generation")
class ReadingPicturesWritingPoems(hub.Module):
    def _initialize(self):
        """
        Initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets",
                                                  "infer_model")
        self.module_image = hub.Module(name="xception71_imagenet") # 调用图像分类的模型
        self.module_similar = hub.Module(name="ernie_gen_couplet") # 调用对联生成的模型
        self.module_poem = hub.Module(name="ernie_gen_poetry")     # 调用古诗生成的模型

    def is_chinese(self, string):
        """
        检查整个字符串是否为中文
        Args:
            string (str): 需要检查的字符串,包含空格也是False
        Return
            bool
        """
        if (len(string) <= 1): # 去除只有单个字或者为空的字符串
            return False

        for chart in string: # 把除了中文的所有字母、数字、符号去除
            if (chart < u'\u4e00' or chart > u'\u9fff'):
                return False

        return True

    def WritingPoem(self, image, use_gpu=False):
        input_dict = {"image": [image]}
        results_image = self.module_image.classification(data=input_dict)
        PictureClassification = list(results_image[0][0].keys())[0]
        translator = Translator(to_lang="chinese")
        PictureClassification_ch = translator.translate("{}".format(PictureClassification))
        texts = ["{}".format(PictureClassification_ch)]
        results_keywords = self.module_similar.generate(texts=texts, use_gpu=use_gpu, beam_width=20)
        Words = [] # 将符合标准的近义词保存在这里（标准：字符串为中文且长度大于1）
        for item in range(20):
            if (self.is_chinese(results_keywords[0][item])):
                Words.append(results_keywords[0][item])
        # 古诗的一句可以拆分成许多词语，因此这里先找到能合成古诗的词语
        FirstWord = Words[0] + Words[1]
        SecondWord = Words[2] + Words[3]
        ThirdWord = Words[4] + Words[5]
        FourthWord = Words[6] + Words[7]
        # 出句和对句，也可以理解为上下句（专业讲法是出句和对句，古诗词是中国传统文化，出句和对句的英文翻译即拼音）
        ChuJu = FirstWord + SecondWord # 出句
        DuiJu = ThirdWord + FourthWord # 对句
        FirstPoetry = ["{:.5}，{:.5}。".format(ChuJu, DuiJu)] # 古诗词的上阕
        results = self.module_poem.generate(texts=FirstPoetry, use_gpu=use_gpu, beam_width=5)
        SecondPoetry = ["{:.12}".format(results[0][0])]
        Poetrys = []
        Poetrys.append(FirstPoetry)
        Poetrys.append(SecondPoetry)
        print("根据图片生成的古诗词：")
        print("{}".format(Poetrys[0][0]))
        print("{}".format(Poetrys[1][0]))
        results = [{
            'image': image,
            'Poetrys': "{}".format(Poetrys[0][0] + Poetrys[1][0])
        }]
        
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

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required.")

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
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU for prediction")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_image',
            type=str,
            default=None,
            help="Pictures to write poetry")
        
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
