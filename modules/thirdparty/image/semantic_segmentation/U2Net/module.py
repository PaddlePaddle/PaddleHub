import os
import paddle
import paddle.nn as nn
import numpy as np
from U2Net.u2net import U2NET
from U2Net.processor import Processor
from paddlehub.module.module import moduleinfo


@moduleinfo(
    name="U2Net",  # 模型名称
    type="CV",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="U2Net",  # 模型介绍
    version="1.0.0"  # 版本号
)
class U2Net(nn.Layer):
    def __init__(self):
        super(U2Net, self).__init__()
        self.model = U2NET(3, 1)
        state_dict = paddle.load(os.path.join(self.directory, 'u2net.pdparams'))
        self.model.set_dict(state_dict)
        self.model.eval()

    def predict(self, input_datas):
        outputs = []
        for data in input_datas:
            data = paddle.to_tensor(data, 'float32')
            d1, d2, d3, d4, d5, d6, d7 = self.model(data)
            outputs.append(d1.numpy())

        outputs = np.concatenate(outputs, 0)

        return outputs

    def Segmentation(self,
                     images=None,
                     paths=None,
                     batch_size=1,
                     input_size=320,
                     output_dir='output',
                     visualization=False):

        # 初始化数据处理器
        processor = Processor(paths, images, batch_size, input_size)

        # 模型预测
        outputs = self.predict(processor.input_datas)

        # 预测结果后处理
        results = processor.postprocess(outputs, visualization=visualization, output_dir=output_dir)

        return results
