# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import copy
import time
import os
import argparse

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from stylepro_artistic.encoder_network import encoder_net
from stylepro_artistic.decoder_network import decoder_net
from stylepro_artistic.processor import postprocess, fr, cv2_to_base64, base64_to_cv2
from stylepro_artistic.data_feed import reader


@moduleinfo(
    name="stylepro_artistic",
    version="1.0.1",
    type="cv/style_transfer",
    summary=
    "StylePro Artistic is an algorithm for Arbitrary image style, which is parameter-free, fast yet effective.",
    author="baidu-bdl",
    author_email="")
class StyleProjection(hub.Module):
    def _initialize(self):
        self.pretrained_encoder_net = os.path.join(self.directory,
                                                   "style_projection_enc")
        self.pretrained_decoder_net = os.path.join(self.directory,
                                                   "style_projection_dec")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        # encoder
        cpu_config_enc = AnalysisConfig(self.pretrained_encoder_net)
        cpu_config_enc.disable_glog_info()
        cpu_config_enc.disable_gpu()
        self.cpu_predictor_enc = create_paddle_predictor(cpu_config_enc)
        # decoder
        cpu_config_dec = AnalysisConfig(self.pretrained_decoder_net)
        cpu_config_dec.disable_glog_info()
        cpu_config_dec.disable_gpu()
        self.cpu_predictor_dec = create_paddle_predictor(cpu_config_dec)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            # encoder
            gpu_config_enc = AnalysisConfig(self.pretrained_encoder_net)
            gpu_config_enc.disable_glog_info()
            gpu_config_enc.enable_use_gpu(
                memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor_enc = create_paddle_predictor(gpu_config_enc)
            # decoder
            gpu_config_dec = AnalysisConfig(self.pretrained_decoder_net)
            gpu_config_dec.disable_glog_info()
            gpu_config_dec.enable_use_gpu(
                memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor_dec = create_paddle_predictor(gpu_config_dec)

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
                    "Attempt to use GPU for prediction, but environment variable CUDA_VISIBLE_DEVICES was not set correctly."
                )

        im_output = []
        for component, w, h in reader(images, paths):
            content = PaddleTensor(component['content_arr'].copy())
            content_feats = self.gpu_predictor_enc.run(
                [content]) if use_gpu else self.cpu_predictor_enc.run([content])
            accumulate = np.zeros((3, 512, 512))
            for idx, style_arr in enumerate(component['styles_arr_list']):
                style = PaddleTensor(style_arr.copy())
                # encode
                style_feats = self.gpu_predictor_enc.run(
                    [style]) if use_gpu else self.cpu_predictor_enc.run([style])
                fr_feats = fr(content_feats[0].as_ndarray(),
                              style_feats[0].as_ndarray(), alpha)
                fr_feats = PaddleTensor(fr_feats.copy())
                # decode
                predict_outputs = self.gpu_predictor_dec.run([
                    fr_feats
                ]) if use_gpu else self.cpu_predictor_dec.run([fr_feats])
                # interpolation
                accumulate += predict_outputs[0].as_ndarray(
                )[0] * component['style_interpolation_weights'][idx]
            # postprocess
            save_im_name = 'ndarray_{}.jpg'.format(time.time())
            result = postprocess(
                accumulate,
                output_dir,
                save_im_name,
                visualization,
                size=(w, h))
            im_output.append(result)
        return im_output

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True):
        encode_dirname = os.path.join(dirname, 'encoder')
        decode_dirname = os.path.join(dirname, 'decoder')
        self._save_encode_model(encode_dirname, model_filename, params_filename,
                                combined)
        self._save_decode_model(decode_dirname, model_filename, params_filename,
                                combined)

    def _save_encode_model(self,
                           dirname,
                           model_filename=None,
                           params_filename=None,
                           combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        encode_program, encode_feeded_var_names, encode_target_vars = fluid.io.load_inference_model(
            dirname=self.pretrained_encoder_net, executor=exe)

        fluid.io.save_inference_model(
            dirname=dirname,
            main_program=encode_program,
            executor=exe,
            feeded_var_names=encode_feeded_var_names,
            target_vars=encode_target_vars,
            model_filename=model_filename,
            params_filename=params_filename)

    def _save_decode_model(self,
                           dirname,
                           model_filename=None,
                           params_filename=None,
                           combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        decode_program, decode_feeded_var_names, decode_target_vars = fluid.io.load_inference_model(
            dirname=self.pretrained_decoder_net, executor=exe)

        fluid.io.save_inference_model(
            dirname=dirname,
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
            image['styles'] = [
                base64_to_cv2(style) for style in image['styles']
            ]
        results = self.style_transfer(images_decode, **kwargs)
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
        if args.weights is None:
            paths = [{
                'content': args.content,
                'styles': args.styles.split(',')
            }]
        else:
            paths = [{
                'content': args.content,
                'styles': args.styles.split(','),
                'weights': list(args.weights)
            }]
        results = self.style_transfer(
            paths=paths,
            alpha=args.alpha,
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=True)
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
            default='transfer_result',
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization',
            type=ast.literal_eval,
            default=True,
            help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--content', type=str, help="path to content.")
        self.arg_input_group.add_argument(
            '--styles', type=str, help="path to styles.")
        self.arg_input_group.add_argument(
            '--weights',
            type=ast.literal_eval,
            default=None,
            help="interpolation weights of styles.")
        self.arg_config_group.add_argument(
            '--alpha',
            type=ast.literal_eval,
            default=1,
            help="The parameter to control the tranform degree.")
