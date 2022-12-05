# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import base64
import os
import time
from functools import reduce
from typing import Union

import numpy as np
import solov2.data_feed as D
import solov2.processor as P

from paddlehub.module.module import moduleinfo
from paddlehub.module.module import serving


class Detector:
    """
    Args:
        min_subgraph_size (int): number of tensorRT graphs.
        use_gpu (bool): whether use gpu
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self, min_subgraph_size: int = 60, use_gpu=False):

        self.default_pretrained_model_path = os.path.join(self.directory, 'solov2_r50_fpn_1x', 'model')
        self.predictor = D.load_predictor(self.default_pretrained_model_path,
                                          min_subgraph_size=min_subgraph_size,
                                          use_gpu=use_gpu)
        self.compose = [
            P.Resize(max_size=1333),
            P.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            P.Permute(),
            P.PadStride(stride=32)
        ]

    def transform(self, im: Union[str, np.ndarray]):
        im, im_info = P.preprocess(im, self.compose)
        inputs = D.create_inputs(im, im_info)
        return inputs, im_info

    def postprocess(self, np_boxes: np.ndarray, np_masks: np.ndarray, threshold: float = 0.5):
        # postprocess output of predictor
        results = {}
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        for box in np_boxes:
            print('class_id:{:d}, confidence:{:.4f},'
                  'left_top:[{:.2f},{:.2f}],'
                  ' right_bottom:[{:.2f},{:.2f}]'.format(int(box[0]), box[1], box[2], box[3], box[4], box[5]))
        results['boxes'] = np_boxes
        if np_masks is not None:
            np_masks = np_masks[expect_boxes, :, :, :]
            results['masks'] = np_masks
        return results

    def predict(self, image: Union[str, np.ndarray], threshold: float = 0.5):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape:[N, class_num, mask_resolution, mask_resolution]
        '''
        inputs, im_info = self.transform(image)
        np_boxes, np_masks = None, None

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        # do not perform postprocess in benchmark mode
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.array([])}
        else:
            results = self.postprocess(np_boxes, np_masks, im_info, threshold=threshold)
        return results


@moduleinfo(name="solov2",
            type="CV/instance_segmentation",
            author="paddlepaddle",
            author_email="",
            summary="solov2 is a detection model, this module is trained with COCO dataset.",
            version="1.2.0")
class DetectorSOLOv2(Detector):
    """
    Args:
        use_gpu (bool): whether use gpu
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self, use_gpu: bool = False):
        super(DetectorSOLOv2, self).__init__(use_gpu=use_gpu)

    def predict(self,
                image: Union[str, np.ndarray],
                threshold: float = 0.5,
                visualization: bool = False,
                save_dir: str = 'solov2_result'):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
            visualization (bool): Whether to save visualization result.
            save_dir (str): save path.

        '''

        inputs, im_info = self.transform(image)
        np_label, np_score, np_segms = None, None, None

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        np_label = self.predictor.get_output_handle(output_names[1]).copy_to_cpu()
        np_score = self.predictor.get_output_handle(output_names[2]).copy_to_cpu()
        np_segms = self.predictor.get_output_handle(output_names[3]).copy_to_cpu()
        output = dict(segm=np_segms, label=np_label, score=np_score)

        if visualization:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image = D.visualize_box_mask(im=image, results=output, threshold=threshold)
            name = str(time.time()) + '.png'
            save_path = os.path.join(save_dir, name)
            image.save(save_path)
        return output

    @serving
    def serving_method(self, images: list, **kwargs):
        """
        Run as a service.
        """
        images_decode = D.base64_to_cv2(images[0])
        results = self.predict(image=images_decode, **kwargs)
        final = {}
        final['segm'] = base64.b64encode(results['segm']).decode('utf8')
        final['label'] = base64.b64encode(results['label']).decode('utf8')
        final['score'] = base64.b64encode(results['score']).decode('utf8')
        return final

    def create_gradio_app(self):
        import os
        import tempfile
        import gradio as gr

        from PIL import Image

        def inference(img, threshold):
            with tempfile.TemporaryDirectory() as tempdir_name:
                self.predict(image=img, threshold=threshold, visualization=True, save_dir=tempdir_name)
                result_names = os.listdir(tempdir_name)
                return Image.open(os.path.join(tempdir_name, result_names[0]))

        interface = gr.Interface(inference,
                                 inputs=[gr.inputs.Image(type="filepath"),
                                         gr.Slider(0.0, 1.0, value=0.5)],
                                 outputs=gr.Image(label='segmentation'),
                                 title='SOLOv2',
                                 allow_flagging='never')

        return interface
