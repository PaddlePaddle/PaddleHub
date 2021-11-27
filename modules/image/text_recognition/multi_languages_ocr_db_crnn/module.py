import argparse
import sys
import os
import ast
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR, draw_ocr
from paddleocr.tools.infer.utility import base64_to_cv2
from paddleocr.ppocr.utils.logging import get_logger
from paddlehub.module.module import moduleinfo, runnable, serving


@moduleinfo(
    name="multi_languages_ocr_db_crnn",
    version="1.0.0",
    summary="ocr service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class MultiLangOCR:
    def __init__(self, lang="ch", det=True, rec=True, use_angle_cls=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        Args:
            lang(str): the selection of languages
            det(bool): Whether to use text detector.
            rec(bool): Whether to use text recognizer.
            use_angle_cls(bool): Whether to use text orientation classifier.
            enable_mkldnn(bool): Whether to enable mkldnn.
        """
        self.enable_mkldnn = enable_mkldnn
        self.logger = get_logger()
        if len(sys.argv) == 1:
            argv = sys.argv
            argc = len(sys.argv)
            print("huangshenghui:::: ", argv, argc)
            self.lang = lang
            self.det = det
            self.rec = rec
            self.use_angle_cls = use_angle_cls
            self.engine = PaddleOCR(
                lang=self.lang,
                det=self.det,
                rec=self.rec,
                use_angle_cls=self.use_angle_cls,
                enable_mkldnn=enable_mkldnn)

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue
            images.append(img)
        return images

    def recognize_text(self,
                       images=[],
                       paths=[],
                       use_gpu=False,
                       output_dir='ocr_result',
                       visualization=False,
                       box_thresh=0.6,
                       angle_classification_thresh=0.9):
        """
        Get the text in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The directory to store output images.
            visualization (bool): Whether to save image or not.
            box_thresh(float): the threshold of the detected text box's confidence
            angle_classification_thresh(float): the threshold of the angle classification confidence
        Returns:
            res (list): The result of text detection box and save path of images.
        """

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."
        self.engine.__init__(det_db_box_thresh=box_thresh, cls_thresh=angle_classification_thresh, use_gpu=use_gpu)
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
                result['save_path'] = self.save_result_image(original_image, rec_results, output_dir)

            all_results.append(result)
        return all_results

    def save_result_image(self, original_image, rec_results, output_dir='ocr_result'):
        image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        if self.det and self.rec:
            boxes = [line[0] for line in rec_results]
            txts = [line[1][0] for line in rec_results]
            scores = [line[1][1] for line in rec_results]
            fonts_lang = 'fonts/simfang.ttf'
            lang_fonts = {
                'korean': 'korean',
                'fr': 'french',
                'german': 'german',
                'hi': 'hindi',
                'ne': 'nepali',
                'fa': 'persian',
                'es': 'spanish',
                'ta': 'tamil',
                'te': 'telugu',
                'ur': 'urdu',
                'ug': 'uyghur',
            }
            if self.lang in lang_fonts.keys():
                fonts_lang = 'fonts/' + lang_fonts[self.lang] + '.ttf'
            font_file = os.path.join(self.directory, 'assets', fonts_lang)
            im_show = draw_ocr(image, boxes, txts, scores, font_path=font_file)
        elif self.det and not self.rec:
            boxes = rec_results
            im_show = self.draw_boxes(image, boxes)
            im_show = np.array(im_show)
        else:
            self.logger.warning("only cls or rec not supported visualization.")
            return ""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ext = self.get_image_ext(original_image)
        saved_name = 'ndarray_{}{}'.format(time.time(), ext)
        save_file_path = os.path.join(output_dir, saved_name)
        im_show = Image.fromarray(im_show)
        im_show.save(save_file_path)
        return save_file_path

    def draw_boxes(self, image, boxes, scores=None, drop_score=0.5):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        if scores is None:
            scores = [1] * len(boxes)
        for (box, score) in zip(boxes, scores):
            if score < drop_score:
                continue
            draw.line([(box[0][0], box[0][1]), (box[1][0], box[1][1])], fill='red')
            draw.line([(box[1][0], box[1][1]), (box[2][0], box[2][1])], fill='red')
            draw.line([(box[2][0], box[2][1]), (box[3][0], box[3][1])], fill='red')
            draw.line([(box[3][0], box[3][1]), (box[0][0], box[0][1])], fill='red')
            draw.line([(box[0][0] - 1, box[0][1] + 1), (box[1][0] - 1, box[1][1] + 1)], fill='red')
            draw.line([(box[1][0] - 1, box[1][1] + 1), (box[2][0] - 1, box[2][1] + 1)], fill='red')
            draw.line([(box[2][0] - 1, box[2][1] + 1), (box[3][0] - 1, box[3][1] + 1)], fill='red')
            draw.line([(box[3][0] - 1, box[3][1] + 1), (box[0][0] - 1, box[0][1] + 1)], fill='red')
        return img

    def get_image_ext(self, image):
        if image.shape[2] == 4:
            return ".png"
        return ".jpg"

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
        self.lang = args.lang
        self.det = args.det
        self.rec = args.rec
        self.use_angle_cls = args.use_angle_cls
        self.engine = PaddleOCR(
            lang=self.lang,
            det=self.det,
            rec=self.rec,
            use_angle_cls=self.use_angle_cls,
            enable_mkldnn=self.enable_mkldnn)
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
        parser.add_argument('--lang', type=str, default='ch', help="the selection of languages")
        parser.add_argument('--det', type=ast.literal_eval, default=True, help="whether use text detector or not")
        parser.add_argument('--rec', type=ast.literal_eval, default=True, help="whether use text recognizer or not")
        parser.add_argument(
            '--use_angle_cls', type=ast.literal_eval, default=False, help="whether text orientation classifier or not")
        return parser
