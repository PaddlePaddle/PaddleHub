# coding: utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import six
import abc


class BaseModuleInfo(object):
    def __init__(self):
        self._modules_info = {}
        self._modules = []

    def set_modules_info(self, modules_info):
        # dict of modules info.
        self._modules_info = modules_info
        # list of modules name.
        self._modules = list(self._modules_info.keys())

    def get_module_info(self, module_name):
        return self._modules_info[module_name]

    def add_module(self, module_name, module_info):
        self._modules_info.update(module_info)
        self._modules.append(module_name)

    def get_module(self, module_name):
        return self.get_module_info(module_name).get("module", None)

    @property
    def modules_info(self):
        return self._modules_info


class CVModuleInfo(BaseModuleInfo):
    def __init__(self):
        self.cv_module_method = {
            "vgg19_imagenet": "predict_classification",
            "vgg16_imagenet": "predict_classification",
            "vgg13_imagenet": "predict_classification",
            "vgg11_imagenet": "predict_classification",
            "shufflenet_v2_imagenet": "predict_classification",
            "se_resnext50_32x4d_imagenet": "predict_classification",
            "se_resnext101_32x4d_imagenet": "predict_classification",
            "resnet_v2_50_imagenet": "predict_classification",
            "resnet_v2_34_imagenet": "predict_classification",
            "resnet_v2_18_imagenet": "predict_classification",
            "resnet_v2_152_imagenet": "predict_classification",
            "resnet_v2_101_imagenet": "predict_classification",
            "pnasnet_imagenet": "predict_classification",
            "nasnet_imagenet": "predict_classification",
            "mobilenet_v2_imagenet": "predict_classification",
            "googlenet_imagenet": "predict_classification",
            "alexnet_imagenet": "predict_classification",
            "yolov3_coco2017": "predict_object_detection",
            "ultra_light_fast_generic_face_detector_1mb_640": "predict_object_detection",
            "ultra_light_fast_generic_face_detector_1mb_320": "predict_object_detection",
            "ssd_mobilenet_v1_pascal": "predict_object_detection",
            "pyramidbox_face_detection": "predict_object_detection",
            "faster_rcnn_coco2017": "predict_object_detection",
            "cyclegan_cityscapes": "predict_gan",
            "deeplabv3p_xception65_humanseg": "predict_semantic_segmentation",
            "ace2p": "predict_semantic_segmentation",
            "pyramidbox_lite_server_mask": "predict_mask",
            "pyramidbox_lite_mobile_mask": "predict_mask"
        }
        super(CVModuleInfo, self).__init__()

    @property
    def cv_modules(self):
        return self._modules

    def add_module(self, module_name, module_info):
        if "CV" == module_info[module_name].get("category", ""):
            self._modules_info.update(module_info)
            self._modules.append(module_name)


class NLPModuleInfo(BaseModuleInfo):
    def __init__(self):
        super(NLPModuleInfo, self).__init__()

    @property
    def nlp_modules(self):
        return self._modules

    def add_module(self, module_name, module_info):
        if "NLP" == module_info[module_name].get("category", ""):
            self._modules_info.update(module_info)
            self._modules.append(module_name)


class V2ModuleInfo(BaseModuleInfo):
    def __init__(self):
        super(V2ModuleInfo, self).__init__()

    @property
    def modules(self):
        return self._modules

    def add_module(self, module_name, module_info):
        self._modules_info.update(module_info)
        self._modules.append(module_name)


class BaseModelService(object):
    def _initialize(self):
        pass

    @abc.abstractmethod
    def _pre_processing(self, data):
        pass

    @abc.abstractmethod
    def _inference(self, data):
        pass

    @abc.abstractmethod
    def _post_processing(self, data):
        pass


cv_module_info = CVModuleInfo()
nlp_module_info = NLPModuleInfo()
v2_module_info = V2ModuleInfo()
