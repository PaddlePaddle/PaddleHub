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
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from Extract_Line_Draft.function import *


@moduleinfo(
    name="Extract_Line_Draft",
    version="1.0.0",
    type="cv/segmentation",
    summary="Import the color picture and generate the line draft of the picture",
    author="彭兆帅，郑博培",
    author_email="1084667371@qq.com，2733821739@qq.com")
class ExtractLineDraft(hub.Module):
    def _initialize(self):
        """
        Initialize with the necessary elements
        """
        # 加载模型路径
        self.default_pretrained_model_path = os.path.join(self.directory, "assets", "infer_model")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        self.model_file_path = self.default_pretrained_model_path
        cpu_config = AnalysisConfig(self.model_file_path)
        cpu_config.disable_glog_info()
        cpu_config.switch_ir_optim(True)
        cpu_config.enable_memory_optim()
        cpu_config.switch_use_feed_fetch_ops(False)
        cpu_config.switch_specify_input_names(True)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_paddle_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = AnalysisConfig(self.model_file_path)
            gpu_config.disable_glog_info()
            gpu_config.switch_ir_optim(True)
            gpu_config.enable_memory_optim()
            gpu_config.switch_use_feed_fetch_ops(False)
            gpu_config.switch_specify_input_names(True)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(100, 0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    # 模型预测函数
    def predict(self, input_datas):
        outputs = []
        # 遍历输入数据进行预测
        for input_data in input_datas:
            inputs = input_data.copy()
            self.input_tensor.copy_from_cpu(inputs)
            self.predictor.zero_copy_run()
            output = self.output_tensor.copy_to_cpu()
            outputs.append(output)

        # 预测结果合并
        outputs = np.concatenate(outputs, 0)

        # 返回预测结果
        return outputs

    def ExtractLine(self, image, use_gpu=False):
        """
        Get the input and program of the infer model

        Args:
             image (list(numpy.ndarray)): images data, shape of each is [H, W, C], the color space is BGR.
             use_gpu(bool): Weather to use gpu
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        from_mat = cv2.imread(image)
        width = float(from_mat.shape[1])
        height = float(from_mat.shape[0])
        new_width = 0
        new_height = 0
        if (width > height):
            from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
            new_width = 512
            new_height = int(512 / width * height)
        else:
            from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
            new_width = int(512 / height * width)
            new_height = 512

        from_mat = from_mat.transpose((2, 0, 1))
        light_map = np.zeros(from_mat.shape, dtype=np.float)
        for channel in range(3):
            light_map[channel] = get_light_map_single(from_mat[channel])
        light_map = normalize_pic(light_map)
        light_map = resize_img_512_3d(light_map)
        light_map = light_map.astype('float32')

        # 获取模型的输入输出
        if use_gpu:
            self.predictor = self.gpu_predictor
        else:
            self.predictor = self.cpu_predictor

        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.input_tensor = self.predictor.get_input_tensor(self.input_names[0])
        self.output_tensor = self.predictor.get_output_tensor(self.output_names[0])
        line_mat = self.predict(np.expand_dims(light_map, axis=0).astype('float32'))
        # 去除 batch 维度 (512, 512, 3)
        line_mat = line_mat.transpose((3, 1, 2, 0))[0]
        # 裁剪 (512, 384, 3)
        line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
        line_mat = np.amax(line_mat, 2)
        # 保存图片
        if Path('./output/').exists():
            show_active_img_and_save_denoise(line_mat, './output/' + 'output.png')
        else:
            os.makedirs('./output/')
            show_active_img_and_save_denoise(line_mat, './output/' + 'output.png')
        print('图片已经完成')

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

        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except RuntimeError:
            self.parser.print_help()
            return None

        use_gpu = args.use_gpu
        self.ExtractLine(image=input_data, use_gpu=use_gpu)

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument('--image', type=str, default=None, help="file contain input data")
        self.arg_input_group.add_argument('--use_gpu', type=ast.literal_eval, default=None, help="weather to use gpu")

    def check_input_data(self, args):
        input_data = []
        if args.image:
            if not os.path.exists(args.image):
                raise RuntimeError("Path %s is not exist." % args.image)
        path = "{}".format(args.image)
        return path
