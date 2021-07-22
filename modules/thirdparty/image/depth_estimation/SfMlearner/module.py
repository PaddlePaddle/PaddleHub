import os
import argparse

import numpy as np
import paddle
from paddlehub.module.module import runnable, moduleinfo, serving
from paddle.vision.transforms import Resize
from PIL import Image

from SfMlearner.net import DispNetS
from SfMlearner.utils import tensor2array


@moduleinfo(
    name="SfMlearner",
    version="1.0.0",
    summary="This is a PaddleHub implementation of SfMlearner.",
    author="JAB",
    author_email="",
    type="cv/depth_estimation",
)
class SfMlearner:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Run the pretrained SfMlearner module.', prog='hub run senta_test', usage='%(prog)s', add_help=True)
        self.parser.add_argument('--paths', default=None, help='paths for different images, list type')
        self.parser.add_argument('--output_dir', default='output/', help='directory for saving depth images')
        self.parser.add_argument('--output_disp', default=True, action='store_false', help='save disparity img')
        self.parser.add_argument('--output_depth', default=False, action='store_true', help='save depth img')
        self.parser.add_argument('--use_gpu', default=False, action='store_true', help='whether to use gpu for prediction')

        self.img_height, self.img_width = 128, 416 

    @serving
    def estimation(self, images=None, paths=None, output_dir='output', use_gpu=False,
                   output_disp=True, output_depth=False):
        """
        Key code for depth estimation

        Args:
            images (list(numpy.ndarray)): Images data. Each element's shape corresponds to [H, W, C]
            paths (list[str]): The paths of images.
            output_dir (str): Output directory for estimation result.
            use_gpu (bool): Whether to use gpu.
            output_disp (bool): Wether to output disparity map.
            output_depth (bool): Whether to output depth map.
        """
        device = paddle.get_device()
        if device == 'cpu' and use_gpu:
            print("You don't have a GPU. Switch to CPU.")
        elif not(device == 'cpu' or use_gpu):
            print("You have a GPU. Recommend using GPU for heavy load prediction.")
            device = 'cpu'
        paddle.set_device(device)
        print('Run prediction on {}.'.format(device))
        pretrained = './SfMlearner/pd_model_trace/model.pdparams'  # path to pretrained parameters
        paddle.disable_static()
        disp_net = DispNetS()
        weights = paddle.load(pretrained)
        disp_net.set_state_dict(weights)  # load pretrained parameters
        disp_net.eval()

        if not(output_disp or output_depth):
            print('You must at least output one value!')
            return
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if (images is not None) and isinstance(images, np.ndarray):  # if images is np.ndarray
            images = [images]
        if (paths is not None) or isinstance(paths, str): # if paths is str
            paths = [paths]
        test_files = []
        if images is not None:
            test_files =['npimg' for _ in range(len(images))]  # Each element in images be labelled as 'npimg'
        if paths is not None:
            test_files = test_files + paths
        print('{} files to test.'.format(len(test_files)))
        print('===============')

        for i, file in enumerate(test_files):
            if file == 'npimg':
                img = images[i]
                file_name, file_ext = 'images'+str(i), 'jpg'
            else:
                img = np.array(Image.open(file))
                file_name, file_ext = os.path.basename(file).split('.')
            img = np.transpose(img, (2, 0, 1))  # 转换为[C, H, W]

            tensor_img_raw = paddle.to_tensor(img.astype(np.float32))
            transform = Resize((self.img_height, self.img_width))
            tensor_img = transform(tensor_img_raw)  # 裁剪为128*416
            tensor_img = ((tensor_img - 127.5)/127.5).unsqueeze(0)

            output = disp_net(tensor_img)[0]  # forward

            if output_disp:
                disp = (255*tensor2array(output, max_value=None, colormap='magma')).astype(np.uint8)
                disp_map = Image.fromarray(np.transpose(disp, (1,2,0)))
                disp_map.save(os.path.join(output_dir, '{}_disp.{}'.format(file_name, file_ext)))
            if output_depth:
                depth = 1/output
                depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                depth_map = Image.fromarray(np.transpose(depth, (1,2,0)))
                depth_map.save(os.path.join(output_dir, '{}_depth.{}'.format(file_name, file_ext)))
        print('All input have been processed.')


    @runnable
    def run_cmd(self, argvs):
        args = self.parser.parse_args(argvs)
        self.estimation(images=None, paths=args.paths, output_dir=args.output_dir,
                        use_gpu=args.use_gpu, output_disp=args.output_disp, output_depth=args.output_depth)