# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import copy
import os

import cv2
import numpy as np
import paddle
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import resize

import paddlehub as hub
from .api import PRN
from .predictor import PosPrediction
from .util import base64_to_cv2
from .utils.render import render_texture
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="prnet", type="CV/", author="paddlepaddle", author_email="", summary="", version="1.0.0")
class PRNet:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "pd_model/model.pdparams")
        self.network = PRN(is_dlib=True, prefix=self.directory)

    def face_swap(self,
                  images: list = None,
                  paths: list = None,
                  mode: int = 0,
                  output_dir: str = './swapping_result/',
                  use_gpu: bool = False,
                  visualization: bool = True):
        '''
        Denoise a raw image in the low-light scene.

        images (list[dict]): data of images, each element is a dict:
          - source (numpy.ndarray): input image，shape is \[H, W, C\]，BGR format；<br/>
          - ref (numpy.ndarray) : style image，shape is \[H, W, C\]，BGR format；<br/>
        paths (list[dict]): paths to images, eacg element is a dict:
          - source (str): path to input image；<br/>
          - ref (str) : path to reference image；<br/>
        mode (int): option, 0 for change part of texture, 1 for change whole face
        output_dir (str): the dir to save the results
        use_gpu (bool): if True, use gpu to perform the computation, otherwise cpu.
        visualization (bool): if True, save results in output_dir.
        '''
        results = []
        paddle.disable_static()
        place = 'gpu:0' if use_gpu else 'cpu'
        place = paddle.set_device(place)
        if images == None and paths == None:
            print('No image provided. Please input an image or a image path.')
            return

        if images != None:
            for image_dict in images:
                source_img = image_dict['source'][:, :, ::-1]
                ref_img = image_dict['ref'][:, :, ::-1]
                results.append(self.texture_editing(source_img, ref_img, mode))

        if paths != None:
            for path_dict in paths:
                source_img = cv2.imread(path_dict['source'])[:, :, ::-1]
                ref_img = cv2.imread(path_dict['ref'])[:, :, ::-1]
                results.append(self.texture_editing(source_img, ref_img, mode))

        if visualization == True:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            for i, out in enumerate(results):
                cv2.imwrite(os.path.join(output_dir, 'output_{}.png'.format(i)), out[:, :, ::-1])

        return results

    def texture_editing(self, source_img, ref_img, mode):
        # read image
        image = source_img
        [h, w, _] = image.shape
        prn = self.network
        #-- 1. 3d reconstruction -> get texture.
        pos = prn.process(image)
        vertices = prn.get_vertices(pos)
        image = image / 255.
        texture = cv2.remap(
            image,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))

        #-- 2. Texture Editing
        Mode = mode
        # change part of texture(for data augumentation/selfie editing. Here modify eyes for example)
        if Mode == 0:
            # load eye mask
            uv_face_eye = imread(os.path.join(self.directory, 'Data/uv-data/uv_face_eyes.png'), as_gray=True) / 255.
            uv_face = imread(os.path.join(self.directory, 'Data/uv-data/uv_face.png'), as_gray=True) / 255.
            eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)

            # texture from another image or a processed texture
            ref_image = ref_img
            ref_pos = prn.process(ref_image)
            ref_image = ref_image / 255.
            ref_texture = cv2.remap(
                ref_image,
                ref_pos[:, :, :2].astype(np.float32),
                None,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0))

            # modify texture
            new_texture = texture * (1 - eye_mask[:, :, np.newaxis]) + ref_texture * eye_mask[:, :, np.newaxis]

        # change whole face(face swap)
        elif Mode == 1:
            # texture from another image or a processed texture
            ref_image = ref_img
            ref_pos = prn.process(ref_image)
            ref_image = ref_image / 255.
            ref_texture = cv2.remap(
                ref_image,
                ref_pos[:, :, :2].astype(np.float32),
                None,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0))
            ref_vertices = prn.get_vertices(ref_pos)
            new_texture = ref_texture  #(texture + ref_texture)/2.

        else:
            print('Wrong Mode! Mode should be 0 or 1.')
            exit()

        #-- 3. remap to input image.(render)
        vis_colors = np.ones((vertices.shape[0], 1))
        face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c=1)
        face_mask = np.squeeze(face_mask > 0).astype(np.float32)

        new_colors = prn.get_colors_from_texture(new_texture)
        new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c=3)
        new_image = image * (1 - face_mask[:, :, np.newaxis]) + new_image * face_mask[:, :, np.newaxis]

        # Possion Editing for blending image
        vis_ind = np.argwhere(face_mask > 0)
        vis_min = np.min(vis_ind, 0)
        vis_max = np.max(vis_ind, 0)
        center = (int((vis_min[1] + vis_max[1]) / 2 + 0.5), int((vis_min[0] + vis_max[0]) / 2 + 0.5))
        output = cv2.seamlessClone((new_image * 255).astype(np.uint8), (image * 255).astype(np.uint8),
                                   (face_mask * 255).astype(np.uint8), center, cv2.NORMAL_CLONE)

        return output

    @runnable
    def run_cmd(self, argvs: list):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        self.args = self.parser.parse_args(argvs)

        self.face_swap(
            paths=[{
                'source': self.args.source,
                'ref': self.args.ref
            }],
            mode=self.args.mode,
            output_dir=self.args.output_dir,
            use_gpu=self.args.use_gpu,
            visualization=self.args.visualization)

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = copy.deepcopy(images)
        for image in images_decode:
            image['source'] = base64_to_cv2(image['source'])
            image['ref'] = base64_to_cv2(image['ref'])
        results = self.face_swap(images_decode, **kwargs)
        tolist = [result.tolist() for result in results]
        return tolist

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--mode', type=int, default=0, help='process option, 0 for part texture, 1 for whole face.', choices=[0, 1])
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='swapping_result', help='output directory for saving result.')
        self.arg_config_group.add_argument('--visualization', type=bool, default=False, help='save results or not.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--source', type=str, help="path to source image.")
        self.arg_input_group.add_argument('--ref', type=str, help="path to reference image.")
