# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import argparse
import ast
import copy
import os
import time

import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from stylepro_artistic.data_feed import reader
from stylepro_artistic.processor import base64_to_cv2
from stylepro_artistic.processor import cv2_to_base64
from stylepro_artistic.processor import fr
from stylepro_artistic.processor import postprocess

import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving
# coding=utf-8


@moduleinfo(
    name="stylepro_artistic",
    version="1.0.3",
    type="cv/style_transfer",
    summary="StylePro Artistic is an algorithm for Arbitrary image style, which is parameter-free, fast yet effective.",
    author="baidu-bdl",
    author_email="")
class StyleProjection(hub.Module):

    def _initialize(self):
        self.pretrained_encoder_net = os.path.join(self.directory, "style_projection_enc")
        self.pretrained_decoder_net = os.path.join(self.directory, "style_projection_dec")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        # encoder
        cpu_config_enc = Config(self.pretrained_encoder_net)
        cpu_config_enc.disable_glog_info()
        cpu_config_enc.disable_gpu()
        self.cpu_predictor_enc = create_predictor(cpu_config_enc)
        # decoder
        cpu_config_dec = Config(self.pretrained_decoder_net)
        cpu_config_dec.disable_glog_info()
        cpu_config_dec.disable_gpu()
        self.cpu_predictor_dec = create_predictor(cpu_config_dec)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            # encoder
            gpu_config_enc = Config(self.pretrained_encoder_net)
            gpu_config_enc.disable_glog_info()
            gpu_config_enc.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor_enc = create_predictor(gpu_config_enc)
            # decoder
            gpu_config_dec = Config(self.pretrained_decoder_net)
            gpu_config_dec.disable_glog_info()
            gpu_config_dec.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor_dec = create_predictor(gpu_config_dec)

    def style_transfer(self,
                       images=None,
                       paths=None,
                       alpha=1,
                       use_gpu=False,
                       output_dir='transfer_result',
                       visualization=False):
        """
        API for image style transfer.

        Args:
            images (list): list of dict objects, each dict contains key:
                content(str): value is a numpy.ndarry with shape [H, W, C], content data.
                styles(str): value is a list of numpy.ndarray with shape [H, W, C], styles data.
                weights(str, optional): value is the interpolation weights correspond to styles.
            paths (list): list of dict objects, each dict contains key:
                content(str): value is the path to content.
                styles(str): value is the paths to styles.
                weights(str, optional): value is the interpolation weights correspond to styles.
            alpha (float): The weight that controls the degree of stylization. Should be between 0 and 1.
            use_gpu (bool): whether to use gpu.
            output_dir (str): the path to store output images.
            visualization (bool): whether to save image or not.

        Returns:
            im_output (list[dict()]): list of output images and save path of images.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        predictor_enc = self.gpu_predictor_enc if use_gpu else self.cpu_predictor_enc
        input_names_enc = predictor_enc.get_input_names()
        input_handle_enc = predictor_enc.get_input_handle(input_names_enc[0])
        output_names_enc = predictor_enc.get_output_names()
        output_handle_enc = predictor_enc.get_output_handle(output_names_enc[0])

        predictor_dec = self.gpu_predictor_dec if use_gpu else self.cpu_predictor_dec
        input_names_dec = predictor_dec.get_input_names()
        input_handle_dec = predictor_dec.get_input_handle(input_names_dec[0])
        output_names_dec = predictor_dec.get_output_names()
        output_handle_dec = predictor_dec.get_output_handle(output_names_dec[0])

        im_output = []
        for component, w, h in reader(images, paths):
            input_handle_enc.copy_from_cpu(component['content_arr'])
            predictor_enc.run()
            content_feats = output_handle_enc.copy_to_cpu()
            accumulate = np.zeros((3, 512, 512))
            for idx, style_arr in enumerate(component['styles_arr_list']):
                # encode
                input_handle_enc.copy_from_cpu(style_arr)
                predictor_enc.run()
                style_feats = output_handle_enc.copy_to_cpu()
                fr_feats = fr(content_feats, style_feats, alpha)
                # decode
                input_handle_dec.copy_from_cpu(fr_feats)
                predictor_dec.run()
                predict_outputs = output_handle_dec.copy_to_cpu()
                # interpolation
                accumulate += predict_outputs[0] * component['style_interpolation_weights'][idx]
            # postprocess
            save_im_name = 'ndarray_{}.jpg'.format(time.time())
            result = postprocess(accumulate, output_dir, save_im_name, visualization, size=(w, h))
            im_output.append(result)
        return im_output

    def save_inference_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        encode_dirname = os.path.join(dirname, 'encoder')
        decode_dirname = os.path.join(dirname, 'decoder')
        self._save_encode_model(encode_dirname, model_filename, params_filename, combined)
        self._save_decode_model(decode_dirname, model_filename, params_filename, combined)

    def _save_encode_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = paddle.CPUPlace()
        exe = paddle.Executor(place)

        encode_program, encode_feeded_var_names, encode_target_vars = paddle.static.load_inference_model(
            dirname=self.pretrained_encoder_net, executor=exe)

        paddle.static.save_inference_model(dirname=dirname,
                                           main_program=encode_program,
                                           executor=exe,
                                           feeded_var_names=encode_feeded_var_names,
                                           target_vars=encode_target_vars,
                                           model_filename=model_filename,
                                           params_filename=params_filename)

    def _save_decode_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = paddle.CPUPlace()
        exe = paddle.Executor(place)

        decode_program, decode_feeded_var_names, decode_target_vars = paddle.static.load_inference_model(
            dirname=self.pretrained_decoder_net, executor=exe)

        paddle.static.save_inference_model(dirname=dirname,
                                           main_program=decode_program,
                                           executor=exe,
                                           feeded_var_names=decode_feeded_var_names,
                                           target_vars=decode_target_vars,
                                           model_filename=model_filename,
                                           params_filename=params_filename)

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = copy.deepcopy(images)
        for image in images_decode:
            image['content'] = base64_to_cv2(image['content'])
            image['styles'] = [base64_to_cv2(style) for style in image['styles']]
        results = self.style_transfer(images_decode, **kwargs)
        results = [{'data': cv2_to_base64(result['data'])} for result in results]
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        if args.weights is None:
            paths = [{'content': args.content, 'styles': args.styles.split(',')}]
        else:
            paths = [{'content': args.content, 'styles': args.styles.split(','), 'weights': list(args.weights)}]
        results = self.style_transfer(paths=paths,
                                      alpha=args.alpha,
                                      use_gpu=args.use_gpu,
                                      output_dir=args.output_dir,
                                      visualization=True)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use GPU or not")
        self.arg_config_group.add_argument('--output_dir',
                                           type=str,
                                           default='transfer_result',
                                           help="The directory to save output images.")
        self.arg_config_group.add_argument('--visualization',
                                           type=ast.literal_eval,
                                           default=True,
                                           help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--content', type=str, help="path to content.")
        self.arg_input_group.add_argument('--styles', type=str, help="path to styles.")
        self.arg_input_group.add_argument('--weights',
                                          type=ast.literal_eval,
                                          default=None,
                                          help="interpolation weights of styles.")
        self.arg_config_group.add_argument('--alpha',
                                           type=ast.literal_eval,
                                           default=1,
                                           help="The parameter to control the tranform degree.")
