# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import os
import argparse
import sys
sys.path.append("..")
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from hrnet_w18_samll_v1_humanseg.processor import postprocess, base64_to_cv2, cv2_to_base64, check_dir
from hrnet_w18_samll_v1_humanseg.data_feed import reader, readtxtpathname


@moduleinfo(
    name="hrnet_w18_samll_v1_humanseg",
    type="CV/semantic_segmentation",
    author="baidu-vis",
    author_email="",
    summary="HRNet_w18_samll_v1 is a semantic segmentation model.",
    version="1.1.1")
class HRNetw18samllv1humanseg(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "humanseg_mobile_inference")
        self._set_config()

    def get_expected_image_width(self):
        return 192

    def get_expected_image_height(self):
        return 192

    def get_pretrained_images_mean(self):
        im_mean = np.array([0.5, 0.5, 0.5]).reshape(1, 3)
        return im_mean

    def get_pretrained_images_std(self):
        im_std = np.array([0.5, 0.5, 0.5]).reshape(1, 3)
        return im_std

    def _set_config(self):
        """
        predictor config setting
        """
        self.model_file_path = os.path.join(self.default_pretrained_model_path,
                                            '__model__')
        self.params_file_path = os.path.join(self.default_pretrained_model_path,
                                             '__params__')
        cpu_config = AnalysisConfig(self.model_file_path, self.params_file_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_paddle_predictor(cpu_config)

    def segmentation(self,
                     images=None,
                     paths=None,
                     data=None,
                     batch_size=1,
                     use_gpu=False,
                     visualization=False,
                     output_dir='humanseg_output'):
        """
        API for human segmentation.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C], the color space is BGR.
            paths (list[str]): The paths of images.
            data (dict): key is 'image', the corresponding value is the path to image.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            visualization (bool): Whether to save image or not.
            output_dir (str): The path to store output images.

        Returns:
            res (list[dict]): each element in the list is a dict, the keys and values are:
                save_path (str, optional): the path to save images. (Exists only if visualization is True)
                data (numpy.ndarray): data of post processed image.
        """

        if use_gpu:
            gpu_config = AnalysisConfig(self.model_file_path,
                                        self.params_file_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(
                memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

        # compatibility with older versions
        if paths:
            paths = readtxtpathname(paths)

        if data and 'image' in data:
            if paths is None:
                paths = list()
            paths += data['image']

        all_data = list()
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = list()
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except:
                    pass
            # feed batch image
            batch_image = np.array([data['image'] for data in batch_data])
            batch_image = PaddleTensor(batch_image.copy())
            output = self.gpu_predictor.run([
                batch_image
            ]) if use_gpu else self.cpu_predictor.run([batch_image])
            output = np.expand_dims(output[0].as_ndarray(), axis=1)
            # postprocess one by one
            for i in range(len(batch_data)):
                out = postprocess(
                    data_out=output[i],
                    org_im=batch_data[i]['org_im'],
                    org_im_shape=batch_data[i]['org_im_shape'],
                    org_im_path=batch_data[i]['org_im_path'],
                    output_dir=output_dir,
                    visualization=visualization)
                res.append(out)
        return res

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        program, feeded_var_names, target_vars = fluid.io.load_inference_model(
            dirname=self.default_pretrained_model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            executor=exe)

        fluid.io.save_inference_model(
            dirname=dirname,
            main_program=program,
            executor=exe,
            feeded_var_names=feeded_var_names,
            target_vars=target_vars,
            model_filename=model_filename,
            params_filename=params_filename)

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.segmentation(images=images_decode, **kwargs)
        results = [{
            'data': cv2_to_base64(result['data'])
        } for result in results]
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
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
        results = self.segmentation(
            paths=args.input_path,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization)
        if args.save_dir is not None:
            check_dir(args.save_dir)
            self.save_inference_model(args.save_dir)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir',
            type=str,
            default='humanseg_output',
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--save_dir',
            type=str,
            default='humanseg_model',
            help="The directory to save model.")
        self.arg_config_group.add_argument(
            '--visualization',
            type=ast.literal_eval,
            default=False,
            help="whether to save output as images.")
        self.arg_config_group.add_argument(
            '--batch_size',
            type=ast.literal_eval,
            default=1,
            help="batch size.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, help="path to image.")


if __name__ == "__main__":
    m = HRNetw18samllv1humanseg()
    import cv2
    img = cv2.imread('检测照片.jpg')
    res = m.segmentation(images=[img], visualization=True)
    print(res[0]['data'])
