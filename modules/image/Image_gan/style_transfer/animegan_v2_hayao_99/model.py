import os

import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor

__all__ = ['InferenceModel']


class InferenceModel:
    # 初始化函数
    def __init__(self, modelpath, use_gpu=False, gpu_id=0, use_mkldnn=False, cpu_threads=1):
        '''
        init the inference model
        modelpath: inference model path
        use_gpu: use gpu or not
        use_mkldnn: use mkldnn or not
        '''
        # 加载模型配置
        self.config = self.load_config(modelpath, use_gpu, gpu_id, use_mkldnn, cpu_threads)

    # 打印函数
    def __repr__(self):
        '''
        get the numbers and name of inputs and outputs
        '''
        return 'input_num: %d\ninput_names: %s\noutput_num: %d\noutput_names: %s' % (
            self.input_num, str(self.input_names), self.output_num, str(self.output_names))

    # 类调用函数
    def __call__(self, *input_datas, batch_size=1):
        '''
        call function
        '''
        return self.forward(*input_datas, batch_size=batch_size)

    # 模型参数加载函数
    def load_config(self, modelpath, use_gpu, gpu_id, use_mkldnn, cpu_threads):
        '''
        load the model config
        modelpath: inference model path
        use_gpu: use gpu or not
        use_mkldnn: use mkldnn or not
        '''
        # 对运行位置进行配置
        if use_gpu:
            try:
                int(os.environ.get('CUDA_VISIBLE_DEVICES'))
            except Exception:
                print(
                    '''Error! Unable to use GPU. Please set the environment variables "CUDA_VISIBLE_DEVICES=GPU_id" to use GPU. Now switch to CPU to continue...'''
                )
                use_gpu = False

        if os.path.isdir(modelpath):
            if os.path.exists(os.path.join(modelpath, "__params__")):
                # __model__ + __params__
                model = os.path.join(modelpath, "__model__")
                params = os.path.join(modelpath, "__params__")
                config = Config(model, params)
            elif os.path.exists(os.path.join(modelpath, "params")):
                # model + params
                model = os.path.join(modelpath, "model")
                params = os.path.join(modelpath, "params")
                config = Config(model, params)
            elif os.path.exists(os.path.join(modelpath, "__model__")):
                # __model__ + others
                config = Config(modelpath)
            else:
                raise Exception("Error! Can\'t find the model in: %s. Please check your model path." %
                                os.path.abspath(modelpath))
        elif os.path.exists(modelpath + ".pdmodel"):
            # *.pdmodel + *.pdiparams
            model = modelpath + ".pdmodel"
            params = modelpath + ".pdiparams"
            config = Config(model, params)
        elif isinstance(modelpath, Config):
            config = modelpath
        else:
            raise Exception("Error! Can\'t find the model in: %s. Please check your model path." %
                            os.path.abspath(modelpath))

        # 设置参数
        if use_gpu:
            config.enable_use_gpu(100, gpu_id)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(cpu_threads)
            if use_mkldnn:
                config.enable_mkldnn()

        config.disable_glog_info()

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

        # 获取模型的输入输出节点数量
        self.input_num = len(self.input_names)
        self.output_num = len(self.output_names)

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
        split_num = datas_num // batch_size + \
                    1 if datas_num % batch_size != 0 else datas_num // batch_size
        input_datas = [np.array_split(input_data, split_num) for input_data in input_datas]

        # 遍历输入数据进行预测
        outputs = {}
        for step in range(split_num):
            for i in range(self.input_num):
                input_data = input_datas[i][step].copy()
                self.input_handles[i].copy_from_cpu(input_data)

            self.predictor.run()

            for i in range(self.output_num):
                output = self.output_handles[i].copy_to_cpu()
                if i in outputs:
                    outputs[i].append(output)
                else:
                    outputs[i] = [output]

        # 预测结果合并
        for key in outputs.keys():
            outputs[key] = np.concatenate(outputs[key], 0)

        outputs = [v for v in outputs.values()]

        # 返回预测结果
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
