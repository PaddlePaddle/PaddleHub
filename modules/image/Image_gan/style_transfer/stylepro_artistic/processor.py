# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import base64
import cv2
import numpy as np

__all__ = ['postprocess', 'fr']


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def postprocess(im, output_dir, save_im_name, visualization, size):
    im = np.multiply(im, 255.0) + 0.5
    im = np.clip(im, 0, 255)
    im = im.astype(np.uint8)
    im = im.transpose((1, 2, 0))
    im = im[:, :, ::-1]
    im = cv2.resize(im, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
    result = {'data': im}
    if visualization:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        elif os.path.isfile(output_dir):
            os.remove(output_dir)
            os.makedirs(output_dir)
        # save image
        save_path = os.path.join(output_dir, save_im_name)
        try:
            cv2.imwrite(save_path, im)
            print('Notice: an image has been proccessed and saved in path "{}".'.format(os.path.abspath(save_path)))
        except Exception as e:
            print('Exception {}: Fail to save output image in path "{}".'.format(e, os.path.abspath(save_path)))
        result['save_path'] = save_path
    return result


def fr(content_feat, style_feat, alpha):
    content_feat = np.reshape(content_feat, (512, -1))
    style_feat = np.reshape(style_feat, (512, -1))

    content_feat_index = np.argsort(content_feat, axis=1)
    style_feat = np.sort(style_feat, axis=1)

    fr_feat = scatter_numpy(dim=1, index=content_feat_index, src=style_feat)
    fr_feat = fr_feat * alpha + content_feat * (1 - alpha)
    fr_feat = np.reshape(fr_feat, (1, 512, 64, 64))
    return fr_feat


def scatter_numpy(dim, index, src):
    """
    Writes all values from the Tensor src into dst at the indices specified in the index Tensor.

    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: dst
    """
    dst = src.copy()
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    dst_xsection_shape = dst.shape[:dim] + dst.shape[dim + 1:]
    if idx_xsection_shape != dst_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output should be the same size")
    if (index >= dst.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and {}.".format(dst.shape[dim] - 1))

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return tuple(slc)

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param.
    idx = [[
        *np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1), index[make_slice(index, dim, i)].reshape(1, -1)[0]
    ] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " + str(dim) +
                             ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        dst[tuple(idx)] = src[tuple(src_idx)]
    else:
        dst[idx] = src
    return dst
