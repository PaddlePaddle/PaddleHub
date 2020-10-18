import argparse
import ast
import os
import math
import six
import base64
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import runnable, serving, moduleinfo
from paddlehub.io.parser import txt_parser
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from translate import Translator
from stgan_bald.net import STGAN_model
from stgan_bald.util.config import add_arguments, print_arguments
from stgan_bald.data_reader import celeba_reader_creator
from stgan_bald.util.utility import check_attribute_conflict, check_gpu, save_batch_image, check_version
from stgan_bald.util import utility
import copy
from PIL import Image

@moduleinfo(
    name="stgan_bald",
    version="1.0.0",
    summary="Baldness generator",
    author="Arrow, 七年期限，Mr.郑先生_",
    author_email="1084667371@qq.com，2733821739@qq.com",
    type="image/gan")
class StganBald(hub.Module):
    def _initialize(self):
        """
        Initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "assets", "infer_model")

    def infer(self, image, use_gpu, output):
        # dataset_dir = "my_dataset"
        data_shape = [None, 3, 128, 128]
        input = fluid.data(name='input', shape=data_shape, dtype='float32')
        label_org_ = fluid.data(
            name='label_org_', shape=[None, 13], dtype='float32')
        label_trg_ = fluid.data(
            name='label_trg_', shape=[None, 13], dtype='float32')
        image_name = fluid.data(
            name='image_name', shape=[None, 1], dtype='int32')

        model_name = 'net_G'

        loader = fluid.io.DataLoader.from_generator(
            feed_list=[input, label_org_, label_trg_, image_name],
            capacity=32,
            iterable=True,
            use_double_buffer=True)
 
        model = STGAN_model()
        fake, _ = model.network_G(
            input,
            label_org_,
            label_trg_,
            name='generator',
            is_test=True)
        

        def _compute_start_end(image_name):
            image_name_start = np.array(image_name)[0].astype('int32')
            image_name_end = image_name_start
            image_name_save = str(np.array(image_name)[0].astype('int32')) + '.jpg'
            # print("read {}.jpg ~ {}.jpg".format(image_name_start, image_name_end))
            return image_name_save

        # prepare environment
        place = fluid.CPUPlace()
        if use_gpu:
            place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        # for var in fluid.default_main_program().all_parameters():
            # print(var.name)

        fluid.load(fluid.default_main_program(), os.path.join("stgan_bald/assets/infer_model", model_name))
        # print('load params done')

        attr_names = "Bald,Bangs,Black_Hair,Blond_Hair,Brown_Hair,Bushy_Eyebrows,Eyeglasses,Male,Mouth_Slightly_Open,Mustache,No_Beard,Pale_Skin,Young".split(',')
        
        test_reader = celeba_reader_creator(
            # image_dir = dataset_dir,
            image = image,
            mode="VAL")
        reader_test = test_reader.make_reader(return_name=True)
        loader.set_batch_generator(
            reader_test,
            places=fluid.cuda_places() if use_gpu else fluid.cpu_places())
        for data in loader():
            real_img, label_org, label_trg, image_name = data[0]['input'], data[
                0]['label_org_'], data[0]['label_trg_'], data[0]['image_name']
            image_name_save = _compute_start_end(image_name)
            real_img_temp = save_batch_image(np.array(real_img))
            ids = "1"
            json = ""
            for i in range(5):
                if (i % 2) == 0:
                    # c_dim必须为13，暂时原因未知，让每次的都为秃头特征处理就好了
                    label_trg_tmp = copy.deepcopy(np.array(label_trg))
                    # images = []
                    images = [real_img_temp]
                    for j in range(len(label_trg_tmp)):
                        new_i = 0 # 代表秃头特征
                        label_trg_tmp[j][new_i] = 1.0 - label_trg_tmp[j][new_i]     # 把测试集标签取反:比如到头发时，取反头发的特征
                        # 刘海
                        # label_trg_tmp[j][new_i+1] = 1.0 - label_trg_tmp[j][new_i+1] # todo
                        # print('check',label_trg_tmp)
                        label_trg_tmp = check_attribute_conflict(
                            label_trg_tmp, attr_names[new_i], attr_names)
                        # print('check——res',label_trg_tmp)
                    # todo：修改变化程度
                    # change_num = 1.5
                    # change_num = i*0.02+0.3
                    change_num = i*0.02+0.3
                    label_org_tmp = list(
                        map(lambda x: ((x * 2) - 1) * change_num, np.array(label_org)))  # 这里的0.5为变化程度，取值0-1，超过1会变形，个人觉得可以取到1.1，看上去变形不是很严重
                    label_trg_tmp = list(
                        map(lambda x: ((x * 2) - 1) * change_num, label_trg_tmp))
                    # 经过测试，如果label_org_tmp和label_trg_tmp内值不一致的取值范围（即变化程度设置不一样），会导致头像特征混乱
                    tensor_label_org_ = fluid.LoDTensor()
                    tensor_label_trg_ = fluid.LoDTensor()
                    tensor_label_org_.set(label_org_tmp, place)
                    tensor_label_trg_.set(label_trg_tmp, place)
                    # print('----',tensor_label_org_,tensor_label_trg_)
                    out = exe.run(feed={
                        "input": real_img,
                        "label_trg_": tensor_label_trg_,
                        "label_org_": tensor_label_org_
                    },fetch_list=[fake.name])
                    fake_temp = save_batch_image(out[0])
                    images.append(fake_temp)
                # 0竖 1横
                    images_concat = np.concatenate(images, 1)
                    if len(np.array(label_org)) > 1:
                        images_concat = np.concatenate(images_concat, 1)
                    fake_image = Image.fromarray(((images_concat + 1) * 127.5).astype(np.uint8))
                    output_name = os.path.join(output, ids + image_name_save)
                    
                    fake_image.save(output_name)
                    with open(output_name,"rb") as f:
                        # b64encode是编码，b64decode是解码
                        base64_data = base64.b64encode(f.read())
                        base64_data = base64_data.decode()
                        # base64.b64decode(base64data)
                        base64_name = "data:image/jpeg;base64," + base64_data
                        json = json + '"image' + ids + '":' + '"' + base64_name + '"' + ','
                    ids = "1" + ids

        json = json[:-1]
        print("{" + json + "}")

    