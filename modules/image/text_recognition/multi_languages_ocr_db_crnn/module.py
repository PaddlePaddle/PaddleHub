import argparse
import sys
import os
import ast

import paddle
import paddle2onnx
import paddle2onnx as p2o
import paddle.fluid as fluid
from paddleocr import PaddleOCR
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.tools.infer.utility import base64_to_cv2
from paddlehub.module.module import moduleinfo, runnable, serving

from .utils import read_images, save_result_image, mkdir


@moduleinfo(
    name="multi_languages_ocr_db_crnn",
    version="1.0.0",
    summary="ocr service",
    author="PaddlePaddle",
    type="cv/text_recognition")
class MultiLangOCR:
    def __init__(self,
                 lang="ch",
                 det=True,
                 rec=True,
                 use_angle_cls=False,
                 enable_mkldnn=False,
                 use_gpu=False,
                 box_thresh=0.6,
                 angle_classification_thresh=0.9):
        """
        initialize with the necessary elements
        Args:
            lang(str): the selection of languages
            det(bool): Whether to use text detector.
            rec(bool): Whether to use text recognizer.
            use_angle_cls(bool): Whether to use text orientation classifier.
            enable_mkldnn(bool): Whether to enable mkldnn.
            use_gpu (bool): Whether to use gpu.
            box_thresh(float): the threshold of the detected text box's confidence
            angle_classification_thresh(float): the threshold of the angle classification confidence
        """
        self.lang = lang
        self.logger = get_logger()
        argc = len(sys.argv)
        if argc == 1 or argc > 1 and sys.argv[1] == 'serving':
            self.det = det
            self.rec = rec
            self.use_angle_cls = use_angle_cls
            self.engine = PaddleOCR(
                lang=lang,
                det=det,
                rec=rec,
                use_angle_cls=use_angle_cls,
                enable_mkldnn=enable_mkldnn,
                use_gpu=use_gpu,
                det_db_box_thresh=box_thresh,
                cls_thresh=angle_classification_thresh)
            self.det_model_dir = self.engine.text_detector.args.det_model_dir
            self.rec_model_dir = self.engine.text_detector.args.rec_model_dir
            self.cls_model_dir = self.engine.text_detector.args.cls_model_dir

    def recognize_text(self, images=[], paths=[], output_dir='ocr_result', visualization=False):
        """
        Get the text in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
            output_dir (str): The directory to store output images.
            visualization (bool): Whether to save image or not.
        Returns:
            res (list): The result of text detection box and save path of images.
        """

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."
        all_results = []
        for img in predicted_data:
            result = {'save_path': ''}
            if img is None:
                result['data'] = []
                all_results.append(result)
                continue
            original_image = img.copy()
            rec_results = self.engine.ocr(img, det=self.det, rec=self.rec, cls=self.use_angle_cls)
            rec_res_final = []
            for line in rec_results:
                if self.det and self.rec:
                    boxes = line[0]
                    text, score = line[1]
                    rec_res_final.append({'text': text, 'confidence': float(score), 'text_box_position': boxes})
                elif self.det and not self.rec:
                    boxes = line
                    rec_res_final.append({'text_box_position': boxes})
                else:
                    if self.use_angle_cls and not self.rec:
                        orientation, score = line
                        rec_res_final.append({'orientation': orientation, 'score': float(score)})
                    else:
                        text, score = line
                        rec_res_final.append({'text': text, 'confidence': float(score)})

            result['data'] = rec_res_final
            if visualization and result['data']:
                result['save_path'] = save_result_image(original_image, rec_results, output_dir, self.directory,
                                                        self.lang, self.det, self.rec, self.logger)

            all_results.append(result)
        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.recognize_text(images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        parser = self.arg_parser()
        args = parser.parse_args(argvs)
        if args.lang is not None:
            self.lang = args.lang
        self.det = args.det
        self.rec = args.rec
        self.use_angle_cls = args.use_angle_cls
        self.engine = PaddleOCR(
            lang=self.lang,
            det=args.det,
            rec=args.rec,
            use_angle_cls=args.use_angle_cls,
            enable_mkldnn=args.enable_mkldnn,
            use_gpu=args.use_gpu,
            det_db_box_thresh=args.box_thresh,
            cls_thresh=args.angle_classification_thresh)
        results = self.recognize_text(
            paths=[args.input_path], output_dir=args.output_dir, visualization=args.visualization)
        return results

    def arg_parser(self):
        parser = argparse.ArgumentParser(
            description="Run the %s module." % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        parser.add_argument('--input_path', type=str, default=None, help="diretory to image. Required.", required=True)
        parser.add_argument('--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU or not")
        parser.add_argument('--output_dir', type=str, default='ocr_result', help="The directory to save output images.")
        parser.add_argument(
            '--visualization', type=ast.literal_eval, default=False, help="whether to save output as images.")
        parser.add_argument('--lang', type=str, default=None, help="the selection of languages")
        parser.add_argument('--det', type=ast.literal_eval, default=True, help="whether use text detector or not")
        parser.add_argument('--rec', type=ast.literal_eval, default=True, help="whether use text recognizer or not")
        parser.add_argument(
            '--use_angle_cls', type=ast.literal_eval, default=False, help="whether text orientation classifier or not")
        parser.add_argument('--enable_mkldnn', type=ast.literal_eval, default=False, help="whether use mkldnn or not")
        parser.add_argument(
            "--box_thresh", type=float, default=0.6, help="set the threshold of the detected text box's confidence")
        parser.add_argument(
            "--angle_classification_thresh",
            type=float,
            default=0.9,
            help="set the threshold of the angle classification confidence")

        return parser

    def export_onnx_model(self, dirname: str, input_shape_dict=None, opset_version=10):
        '''
        Export the model to ONNX format.

        Args:
            dirname(str): The directory to save the onnx model.
            input_shape_dict: dictionary ``{ input_name: input_value }, eg. {'x': [-1, 3, -1, -1]}``
            opset_version(int): operator set
        '''
        v0, v1, v2 = paddle2onnx.__version__.split('.')
        if int(v1) < 9:
            raise ImportError("paddle2onnx>=0.9.0 is required")

        if input_shape_dict is not None and not isinstance(input_shape_dict, dict):
            raise Exception("input_shape_dict should be dict, eg. {'x': [-1, 3, -1, -1]}.")

        if opset_version <= 9:
            raise Exception("opset_version <= 9 is not surpported, please try with higher opset_version >=10.")

        path_dict = {"det": self.det_model_dir, "rec": self.rec_model_dir, "cls": self.cls_model_dir}
        for (key, path) in path_dict.items():
            model_filename = 'inference.pdmodel'
            params_filename = 'inference.pdiparams'
            save_file = os.path.join(dirname, '{}_{}.onnx'.format(self.name, key))

            # convert model save with 'paddle.fluid.io.save_inference_model'
            if hasattr(paddle, 'enable_static'):
                paddle.enable_static()
            exe = fluid.Executor(fluid.CPUPlace())
            if model_filename is None and params_filename is None:
                [program, feed_var_names, fetch_vars] = fluid.io.load_inference_model(path, exe)
            else:
                [program, feed_var_names, fetch_vars] = fluid.io.load_inference_model(
                    path, exe, model_filename=model_filename, params_filename=params_filename)

            onnx_proto = p2o.run_convert(program, input_shape_dict=input_shape_dict, opset_version=opset_version)
            mkdir(save_file)
            with open(save_file, "wb") as f:
                f.write(onnx_proto.SerializeToString())
