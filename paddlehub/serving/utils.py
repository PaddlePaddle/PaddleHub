# coding: utf-8
import hashlib
import multiprocessing
from paddlehub.serving.model_service.text_model_service import TextModelService
from paddlehub.serving.model_service.image_model_service import ImageModelService
import time


def gen_md5(src_str):
    md5_get = hashlib.md5()
    md5_src = str(time.time()) + str(src_str)
    md5_get.update(md5_src.encode("utf-8"))
    md5_value = md5_get.hexdigest()
    return md5_value


def _predict_text(input_data, module_name, use_gpu=False):
    module = TextModelService.get_module(module_name)
    method_name = module.desc.attr.map.data['default_signature'].s
    if method_name != "":
        predict_method = getattr(module, method_name)
        try:
            print("Use gpu is", use_gpu)
            results = predict_method(data=input_data, use_gpu=use_gpu)
        except Exception as err:
            print(err)
            return {"result": "请检查数据格式!"}
    else:
        results = "您调用的模型{}不支持在线预测".format(module_name)

    if results == []:
        results = "输入不能为空"
    return {"result": results}
