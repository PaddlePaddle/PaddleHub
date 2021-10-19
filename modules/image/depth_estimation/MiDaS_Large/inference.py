import os
import numpy as np

from paddle.inference import create_predictor, Config

__all__ = ['InferenceModel']


class InferenceModel():
    # 初始化函数
    def __init__(self, modelpath, use_gpu=False, use_mkldnn=False, combined=True):
        '''
        init the inference model

        modelpath: inference model path

        use_gpu: use gpu or not

        use_mkldnn: use mkldnn or not

        combined: inference model format is combined or not
        '''
        # 加载模型配置
        self.config = self.load_config(modelpath, use_gpu, use_mkldnn, combined)

    # 打印函数
    def __repr__(self):
        '''
        get the numbers and name of inputs and outputs
        '''
        return 'inputs_num: %d\ninputs_names: %s\noutputs_num: %d\noutputs_names: %s' % (len(
            self.input_handles), str(self.input_names), len(self.output_handles), str(self.output_names))

    # 类调用函数
    def __call__(self, *input_datas, batch_size=1):
        '''
        call function
        '''
        return self.forward(*input_datas, batch_size=batch_size)

    # 模型参数加载函数
    def load_config(self, modelpath, use_gpu, use_mkldnn, combined):
        '''
        load the model config

        modelpath: inference model path

        use_gpu: use gpu or not

        use_mkldnn: use mkldnn or not

        combined: inference model format is combined or not
        '''
        # 对运行位置进行配置
        if use_gpu:
            try:
                int(os.environ.get('CUDA_VISIBLE_DEVICES'))
            except Exception:
                print(
                    'Error! Unable to use GPU. Please set the environment variables "CUDA_VISIBLE_DEVICES=GPU_id" to use GPU.'
                )
                use_gpu = False

        # 加载模型参数
        if combined:
            model = os.path.join(modelpath, "__model__")
            params = os.path.join(modelpath, "__params__")
            config = Config(model, params)
        else:
            config = Config(modelpath)

        # 设置参数
        if use_gpu:
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
            if use_mkldnn:
                config.enable_mkldnn()

        # 返回配置
        return config

    # 预测器创建函数
    def eval(self):
        '''
        create the model predictor by model config
        '''
        # 创建预测器
        self.predictor = create_predictor(self.config)

        # 获取模型的输入输出名称
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

        # 获取输入
        self.input_handles = []
        for input_name in self.input_names:
            self.input_handles.append(self.predictor.get_input_handle(input_name))

        # 获取输出
        self.output_handles = []
        for output_name in self.output_names:
            self.output_handles.append(self.predictor.get_output_handle(output_name))

    # 前向计算函数
    def forward(self, *input_datas, batch_size=1):
        """
        model inference

        batch_size: batch size

        *input_datas: x1, x2, ..., xn
        """
        # 切分输入数据
        datas_num = input_datas[0].shape[0]
        split_num = datas_num // batch_size + 1 if datas_num % batch_size != 0 else datas_num // batch_size
        input_datas = [np.array_split(input_data, split_num) for input_data in input_datas]

        # 遍历输入数据进行预测
        outputs = {}
        for step in range(split_num):
            for i in range(len(self.input_handles)):
                input_data = input_datas[i][step].copy()
                self.input_handles[i].copy_from_cpu(input_data)

            self.predictor.run()

            for i in range(len(self.output_handles)):
                output = self.output_handles[i].copy_to_cpu()
                if i in outputs:
                    outputs[i].append(output)
                else:
                    outputs[i] = [output]

        # 预测结果合并
        for key in outputs.keys():
            outputs[key] = np.concatenate(outputs[key], 0)

        # 返回预测结果
        return outputs
