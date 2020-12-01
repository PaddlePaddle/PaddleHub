import os
import sys
import base64

import cv2
import numpy as np
import paddle
import paddle.nn as nn
from PIL import Image


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def is_listy(x):
    return isinstance(x, (tuple, list))


class Hook():
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m, hook_func, is_forward=True, detach=True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_post_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (o.detach() for o in input) if is_listy(input) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward=True, detach=True):
        self.hooks = []
        try:
            for m in ms:
                self.hooks.append(Hook(m, hook_func, is_forward, detach))
        except Exception as e:
            pass

    def __getitem__(self, i: int) -> Hook:
        return self.hooks[i]

    def __len__(self) -> int:
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def _hook_inner(m, i, o):
    return o if isinstance(o, paddle.fluid.framework.Variable) else o if is_listy(o) else list(o)


def hook_output(module, detach=True, grad=False):
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)


def hook_outputs(modules, detach=True, grad=False):
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)


def model_sizes(m, size=(64, 64)):
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]


def dummy_eval(m, size=(64, 64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    m.eval()
    return m(dummy_batch(size))


def dummy_batch(size=(64, 64), ch_in=3):
    "Create a dummy batch to go through `m` with `size`."
    arr = np.random.rand(1, ch_in, *size).astype('float32') * 2 - 1
    return paddle.to_tensor(arr)


class _SpectralNorm(nn.SpectralNorm):
    def __init__(self, weight_shape, dim=0, power_iters=1, eps=1e-12, dtype='float32'):
        super(_SpectralNorm, self).__init__(weight_shape, dim, power_iters, eps, dtype)

    def forward(self, weight):
        inputs = {'Weight': weight, 'U': self.weight_u, 'V': self.weight_v}
        out = self._helper.create_variable_for_type_inference(self._dtype)
        _power_iters = self._power_iters if self.training else 0
        self._helper.append_op(
            type="spectral_norm",
            inputs=inputs,
            outputs={
                "Out": out,
            },
            attrs={
                "dim": self._dim,
                "power_iters": _power_iters,
                "eps": self._eps,
            })

        return out


class Spectralnorm(paddle.nn.Layer):
    def __init__(self, layer, dim=0, power_iters=1, eps=1e-12, dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = _SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


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
        ' -r ', r, ' -f ', ' image2 ', ' -i ', frame_path, ' -vcodec ', ' libx264 ', ' -pix_fmt ', ' yuv420p ',
        ' -crf ', ' 16 ', video_path
    ]
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
