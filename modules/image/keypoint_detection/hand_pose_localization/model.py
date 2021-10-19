import os
import numpy as np

from paddle.inference import create_predictor, Config

__all__ = ['Model']


class Model():
    # 初始化函数
    def __init__(self, modelpath, use_gpu=False, use_mkldnn=True, combined=True, use_device=None):
        # 加载模型预测器
        self.predictor = self.load_model(modelpath, use_gpu, use_mkldnn, combined, use_device)

        # 获取模型的输入输出
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.input_handle = self.predictor.get_input_handle(self.input_names[0])
        self.output_handle = self.predictor.get_output_handle(self.output_names[0])

    def _get_device_id(self, places):
        try:
            places = os.environ[places]
            id = int(places)
        except:
            id = -1
        return id

    # 模型加载函数
    def load_model(self, modelpath, use_gpu, use_mkldnn, combined, use_device):
        # 加载模型参数
        if combined:
            model = os.path.join(modelpath, "__model__")
            params = os.path.join(modelpath, "__params__")
            config = Config(model, params)
        else:
            config = Config(modelpath)

        # 对运行位置进行配置
        if use_device is not None:
            if use_device == "cpu":
                if use_mkldnn:
                    config.enable_mkldnn()
            elif use_device == "xpu":
                xpu_id = self._get_device_id("XPU_VISIBLE_DEVICES")
                if xpu_id != -1:
                    config.enable_xpu(100)
                else:
                    print(
                        'Error! Unable to use XPU. Please set the environment variables "XPU_VISIBLE_DEVICES=XPU_id" to use XPU.'
                    )
            elif use_device == "npu":
                npu_id = self._get_device_id("FLAGS_selected_npus")
                if npu_id != -1:
                    config.enable_npu(device_id=npu_id)
                else:
                    print(
                        'Error! Unable to use NPU. Please set the environment variables "FLAGS_selected_npus=NPU_id" to use NPU.'
                    )
            elif use_device == "gpu":
                gpu_id = self._get_device_id("CUDA_VISIBLE_DEVICES")
                if gpu_id != -1:
                    config.enable_use_gpu(100, gpu_id)
                else:
                    print(
                        'Error! Unable to use GPU. Please set the environment variables "CUDA_VISIBLE_DEVICES=GPU_id" to use GPU.'
                    )
            else:
                raise Exception("Unsupported device: " + use_device)
        else:
            if use_gpu:
                gpu_id = self._get_device_id("CUDA_VISIBLE_DEVICES")
                if gpu_id != -1:
                    config.enable_use_gpu(100, gpu_id)
                else:
                    print(
                        'Error! Unable to use GPU. Please set the environment variables "CUDA_VISIBLE_DEVICES=GPU_id" to use GPU.'
                    )
            else:
                if use_mkldnn:
                    config.enable_mkldnn()

        config.disable_glog_info()
        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        config.switch_specify_input_names(True)

        # 通过参数加载模型预测器
        predictor = create_predictor(config)

        # 返回预测器
        return predictor

    # 模型预测函数
    def predict(self, input_datas):
        outputs = []

        # 遍历输入数据进行预测
        for input_data in input_datas:
            inputs = input_data.copy()
            self.input_handle.copy_from_cpu(inputs)
            self.predictor.run()
            output = self.output_handle.copy_to_cpu()
            outputs.append(output)

        # 预测结果合并
        outputs = np.concatenate(outputs, 0)

        # 返回预测结果
        return outputs
