import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from paddleocr import draw_ocr


def save_result_image(original_image,
                      rec_results,
                      output_dir='ocr_result',
                      directory=None,
                      lang='ch',
                      det=True,
                      rec=True,
                      logger=None):
    image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    if det and rec:
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
        if lang in lang_fonts.keys():
            fonts_lang = 'fonts/' + lang_fonts[lang] + '.ttf'
        font_file = os.path.join(directory, 'assets', fonts_lang)
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_file)
    elif det and not rec:
        boxes = rec_results
        im_show = draw_boxes(image, boxes)
        im_show = np.array(im_show)
    else:
        logger.warning("only cls or rec not supported visualization.")
        return ""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ext = get_image_ext(original_image)
    saved_name = 'ndarray_{}{}'.format(time.time(), ext)
    save_file_path = os.path.join(output_dir, saved_name)
    im_show = Image.fromarray(im_show)
    im_show.save(save_file_path)
    return save_file_path


def read_images(paths=[]):
    images = []
    for img_path in paths:
        assert os.path.isfile(img_path), "The {} isn't a valid file.".format(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        images.append(img)
    return images


def draw_boxes(image, boxes, scores=None, drop_score=0.5):
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


def get_image_ext(image):
    if image.shape[2] == 4:
        return ".png"
    return ".jpg"


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
