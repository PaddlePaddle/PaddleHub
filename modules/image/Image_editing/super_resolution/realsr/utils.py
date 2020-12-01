import os
import sys
import base64

import cv2
from PIL import Image
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
    cmd = ffmpeg + [' -r ', r, ' -f ', ' image2 ', ' -i ', frame_path, ' -pix_fmt ', ' yuv420p ', video_path]
    cmd = ''.join(cmd)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(video_path))

    sys.stdout.flush()


def is_image(input):
    try:
        img = Image.open(input)
        _ = img.size
        return True
    except:
        return False


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data
