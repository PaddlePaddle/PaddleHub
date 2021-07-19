import paddle
from paddlehub.module.module import runnable, moduleinfo
from paddle.vision.transforms import Resize

from PIL import Image
import numpy as np
import os
import argparse
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
        # add arg parser
        self.parser = argparse.ArgumentParser(
            description='Run the pretrained SfMlearner module.', prog='hub run senta_test', usage='%(prog)s', add_help=True)
        self.parser.add_argument('--pretrained', default='SfMlearner/pd_model_trace/model.pdparams', help='pretrained DispNet path')
        self.parser.add_argument('--img-dir', default='input/', help='directory incorporating all input images')
        self.parser.add_argument('--output-dir', default='output/', help='directory for saving depth images')
        self.parser.add_argument('--output-disp', action='store_true', help='save disparity img', default=True)
        self.parser.add_argument('--output-depth', action='store_true', help='save depth img', default=False)
        self.parser.add_argument('--USE-GPU', action='store_true', help='whether to use gpu for prediction', default=False)

        self.img_height, self.img_width = 128, 416 

    def estimation(self, pretrained, img_dir, output_dir, USE_GPU, output_disp=True, output_depth=False):
        device = paddle.get_device()
        if device == 'cpu' and USE_GPU:
            print("You don't have a GPU. Switch to CPU.")
        elif not(device == 'cpu' or USE_GPU):
            print("You have a GPU. Recommend using GPU for heavy load prediction.")
            device = 'cpu'
        paddle.set_device(device)
        print('Run prediction on {}.'.format(device))

        if not(output_disp or output_depth):
            print('You must at least output one value!')
            return

        paddle.disable_static()
        disp_net = DispNetS()
        weights = paddle.load(pretrained)
        disp_net.set_state_dict(weights)  #加载预训练参数
        disp_net.eval()

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        test_files = [img_dir+file for file in os.listdir(img_dir)]
        print('{} files to test.'.format(len(test_files)))
        print('===============')

        for file in test_files:
            img = np.array(Image.open(file))  # 读入图片
            img = np.transpose(img, (2, 0, 1))

            tensor_img_raw = paddle.to_tensor(img.astype(np.float32))
            transform = Resize((self.img_height, self.img_width))
            tensor_img = transform(tensor_img_raw)  # 裁剪为128*416
            tensor_img = ((tensor_img - 127.5)/127.5).unsqueeze(0)

            output = disp_net(tensor_img)[0]  # 前向计算

            file_name, file_ext = os.path.relpath(file, img_dir).split('.')
            if output_disp:
                disp = (255*tensor2array(output, max_value=None, colormap='magma')).astype(np.uint8)
                disp_map = Image.fromarray(np.transpose(disp, (1,2,0)))
                disp_map.save(output_dir+'{}_disp.{}'.format(file_name, file_ext))
            if output_depth:
                depth = 1/output
                depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                depth_map = Image.fromarray(np.transpose(depth, (1,2,0)))
                depth_map.save(output_dir+'{}_depth.{}'.format(file_name, file_ext))
        print('All input have been processed.')


    @runnable
    def run_cmd(self, argvs):
        args = self.parser.parse_args(argvs)
        self.estimation(args.img_dir, args.output_dir, args.USE_GPU,
                        outpput_disp=args.output_disp, output_depth=args.output_depth)