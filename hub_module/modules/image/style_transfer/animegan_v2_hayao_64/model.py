import os
import numpy as np

from paddle.fluid.core import AnalysisConfig, create_paddle_predictor

__all__ = ['Model']

class Model():
    # 初始化函数
    def __init__(self, modelpath, use_gpu):
        # 加载模型预测器
        self.predictor = self.load_model(modelpath, use_gpu)

        # 获取模型的输入输出
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        self.input_tensor = self.predictor.get_input_tensor(self.input_names[0])
        self.output_tensor = self.predictor.get_output_tensor(self.output_names[0])

    # 模型加载函数
    def load_model(self, modelpath, use_gpu):
        # 对运行位置进行配置
        if use_gpu:
            try:
                places = os.environ["CUDA_VISIBLE_DEVICES"]
                places = int(places[0])
            except Exception as e:
                print('Error: %s. Please set the environment variables "CUDA_VISIBLE_DEVICES".' % e)
                use_gpu = False

        # 加载模型参数
        config = AnalysisConfig(modelpath)

        # 设置参数
        if use_gpu:   
            config.enable_use_gpu(100, places)
        else:
            config.disable_gpu()
            config.enable_mkldnn()
        config.disable_glog_info()
        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        config.switch_specify_input_names(True)
        
        # 通过参数加载模型预测器
        predictor = create_paddle_predictor(config)
        
        # 返回预测器
        return predictor

    # 模型预测函数
    def predict(self, input_datas):
        outputs = []

        # 遍历输入数据进行预测
        for input_data in input_datas:
            self.input_tensor.copy_from_cpu(input_data)
            self.predictor.zero_copy_run()
            output = self.output_tensor.copy_to_cpu()
            outputs.append(output)
        
        # 预测结果合并
        outputs = np.concatenate(outputs, 0)

        # 返回预测结果
        return outputs