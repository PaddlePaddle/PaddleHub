# coding=utf-8
import os

import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    'get_save_image_name', 'draw_bounding_box_on_image', 'clip_bbox',
    'load_label_info'
]


def get_save_image_name(img, output_dir, image_path):
    """Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)

    if ext == '':
        if img.format == 'PNG':
            ext = '.png'
        elif img.format == 'JPEG':
            ext = '.jpg'
        elif img.format == 'BMP':
            ext = '.bmp'
        else:
            if img.mode == "RGB" or img.mode == "L":
                ext = ".jpg"
            elif img.mode == "RGBA" or img.mode == "P":
                ext = '.png'

    return os.path.join(output_dir, "{}".format(name)) + ext


def draw_bounding_box_on_image(image_path, data_list, save_dir):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for data in data_list:
        left, right, top, bottom = data['left'], data['right'], data[
            'top'], data['bottom']

        # draw bbox
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=2,
                  fill='red')

        # draw label
        if image.mode == 'RGB':
            text = data['label'] + ": %.2f%%" % (100 * data['confidence'])
            textsize_width, textsize_height = draw.textsize(text=text)
            draw.rectangle(
                xy=(left, top - (textsize_height + 5),
                    left + textsize_width + 10, top),
                fill=(255, 255, 255))
            draw.text(xy=(left, top - 15), text=text, fill=(0, 0, 0))

    save_name = get_save_image_name(image, save_dir, image_path)
    if os.path.exists(save_name):
        os.remove(save_name)

    image.save(save_name)

    return save_name


def clip_bbox(bbox, img_width, img_height):
    xmin = max(min(bbox[0], img_width), 0.)
    ymin = max(min(bbox[1], img_height), 0.)
    xmax = max(min(bbox[2], img_width), 0.)
    ymax = max(min(bbox[3], img_height), 0.)
    return xmin, ymin, xmax, ymax


def load_label_info(file_path):
    with open(file_path, 'r') as fr:
        text = fr.readlines()
        label_names = []
        for info in text:
            label_names.append(info.strip())
        return label_names


def postprocess(paths,
                images,
                data_out,
                score_thresh,
                label_names,
                output_dir,
                handle_id,
                visualization=True):
    """postprocess the lod_tensor produced by fluid.Executor.run

    :param paths: the path of images.
    :type paths: list, each element is a str
    :param images: data of images, [N, H, W, C]
    :type images: numpy.ndarray
    :param data_out: data produced by executor.run
    :type data_out: lod_tensor
    :param score_thresh: the low limit of bounding box.
    :type score_thresh: float
    :param label_names: label names
    :type label_names: list
    :param output_dir: output directory.
    :type output_dir: str
    :param handle_id: The number of images that have been handled.
    :type handle_id: int
    :param visualization: whether to draw bbox.
    :param visualization: bool
    """
    lod_tensor = data_out[0]
    lod = lod_tensor.lod[0]
    results = lod_tensor.as_ndarray()
    if handle_id < len(paths):
        unhandled_paths = paths[handle_id:]
        unhandled_paths_num = len(unhandled_paths)
    else:
        unhandled_paths_num = 0

    output = []
    for index in range(len(lod) - 1):
        output_i = {'data': []}
        if index < unhandled_paths_num:
            org_img_path = unhandled_paths[index]
            org_img = Image.open(org_img_path)
            output_i['path'] = org_img_path
        else:
            org_img = images[index - unhandled_paths_num]
            org_img = org_img.astype(np.uint8)
            org_img = Image.fromarray(org_img[:, :, ::-1])
            if visualization:
                org_img_path = get_save_image_name(
                    org_img, output_dir, 'image_numpy_{}'.format(
                        (handle_id + index)))
                org_img.save(org_img_path)
        org_img_height = org_img.height
        org_img_width = org_img.width
        result_i = results[lod[index]:lod[index + 1]]
        for row in result_i:
            if len(row) != 6:
                continue
            if row[1] < score_thresh:
                continue
            category_id = int(row[0])
            confidence = row[1]
            bbox = row[2:]
            bbox[0] = bbox[0] * org_img_width
            bbox[1] = bbox[1] * org_img_height
            bbox[2] = bbox[2] * org_img_width
            bbox[3] = bbox[3] * org_img_height
            dt = {}
            dt['label'] = label_names[category_id]
            dt['confidence'] = confidence
            dt['left'], dt['top'], dt['right'], dt['bottom'] = clip_bbox(
                bbox, org_img_width, org_img_height)
            output_i['data'].append(dt)

        output.append(output_i)
        if visualization:
            output_i['save_path'] = draw_bounding_box_on_image(
                org_img_path, output_i['data'], output_dir)

    return output
