import os
import cv2
import base64
import numpy as np
import paddle.nn as nn
from paddleocr import PaddleOCR
from paddlehub.module.module import moduleinfo, serving


@moduleinfo(
    name="Vehicle_License_Plate_Recognition",
    type="CV/text_recognition",
    author="jm12138",
    author_email="",
    summary="Vehicle_License_Plate_Recognition",
    version="1.0.0")
class Vehicle_License_Plate_Recognition(nn.Layer):
    def __init__(self):
        super(Vehicle_License_Plate_Recognition, self).__init__()
        self.vlpr = PaddleOCR(
            det_model_dir=os.path.join(self.directory, 'det_vlpr'),
            rec_model_dir=os.path.join(self.directory, 'ch_ppocr_server_v2.0_rec_infer'))

    @staticmethod
    def base64_to_cv2(b64str):
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.frombuffer(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

    def plate_recognition(self, images=None):
        assert isinstance(images, (list, str, np.ndarray))
        results = []

        if isinstance(images, list):
            for item in images:
                for bbox, text in self.vlpr.ocr(item):
                    results.append({'license': text[0], 'bbox': bbox})

        elif isinstance(images, (str, np.ndarray)):
            for bbox, text in self.vlpr.ocr(images):
                results.append({'license': text[0], 'bbox': bbox})

        return results

    @serving
    def serving_method(self, images):
        if isinstance(images, list):
            images_decode = [self.base64_to_cv2(image) for image in images]
        elif isinstance(images, str):
            images_decode = self.base64_to_cv2(images)

        return self.plate_recognition(images_decode)
