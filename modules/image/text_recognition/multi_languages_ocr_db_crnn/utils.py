import os

import cv2
from PIL import ImageDraw


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
