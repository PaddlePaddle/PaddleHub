import os
import sys
import base64

import cv2
import numpy as np


def video2frames(video_path, outpath, **kargs):
    def _dict2str(kargs):
        cmd_str = ''
        for k, v in kargs.items():
            cmd_str += (' ' + str(k) + ' ' + str(v))
        return cmd_str

    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    vid_name = video_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(outpath, vid_name)

    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)

    # video file name
    outformat = out_full_path + '/%08d.png'

    cmd = ffmpeg
    cmd = ffmpeg + [' -i ', video_path, ' -start_number ', ' 0 ', outformat]

    cmd = ''.join(cmd) + _dict2str(kargs)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(vid_name))

    sys.stdout.flush()
    return out_full_path


def frames2video(frame_path, video_path, r):
    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    cmd = ffmpeg + [
        ' -r ', r, ' -f ', ' image2 ', ' -i ', frame_path, ' -vcodec ',
        ' libx264 ', ' -pix_fmt ', ' yuv420p ', ' -crf ', ' 16 ', video_path
    ]
    cmd = ''.join(cmd)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(video_path))

    sys.stdout.flush()


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def get_img(pred):
    pred = pred.squeeze()
    pred = np.clip(pred, a_min=0., a_max=1.0)
    pred = pred * 255
    pred = pred.round()
    pred = pred.astype('uint8')
    pred = np.transpose(pred, (1, 2, 0))  # chw -> hwc
    pred = pred[:, :, ::-1]  # rgb -> bgr
    return pred


def save_img(img, framename):
    dirname = os.path.dirname(framename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    cv2.imwrite(framename, img)


def read_img(path, size=None, is_gt=False):
    """read image by cv2
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def get_test_neighbor_frames(crt_i, N, max_n, padding='new_info'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)

    return return_l