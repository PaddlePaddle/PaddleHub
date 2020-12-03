import os
import paddle
import numpy as np
from paddle.nn import Layer
from paddlehub.module.module import moduleinfo
from CPM_LM.GPT2 import GPT2Model, GPT2Tokenizer

@moduleinfo(
    name="CPM_LM", # 模型名称
    type="NLP/NLG", # 模型类型
    author="jm12138", # 作者名称
    author_email="jm12138@qq.com", # 作者邮箱
    summary="CPM_LM", # 模型介绍
    version="1.0.0" # 版本号
)
class CPM_LM(Layer):
    def __init__(self, max_len=512):
        super(CPM_LM, self).__init__()
        # 初始化模型
        self.model = GPT2Model(
            vocab_size=30000,
            layer_size=32,
            block_size=1024,
            embedding_dropout=0.0,
            embedding_size=2560,
            num_attention_heads=32,
            attention_dropout=0.0,
            residual_dropout=0.0)

        # 读取CPM-LM模型参数(FP16)
        state_dict = paddle.load(os.path.join(self.directory, 'CPM-LM.pdparams'))

        # FP16 -> FP32
        for param in state_dict:
            state_dict[param] = state_dict[param].astype('float32')

        # 加载CPM-LM模型参数
        self.model.set_dict(state_dict)

        # 将模型设置为评估状态
        self.model.eval()

        # 加载编码器
        self.tokenizer = GPT2Tokenizer(
            vocab_file=os.path.join(self.directory, 'GPT2/bpe/vocab.json'),
            model_file=os.path.join(self.directory, 'GPT2/bpe/chinese_vocab.model'),
            max_len=max_len)

        # 初始化编码器
        _ = self.tokenizer.encode('_')

    # 基础预测函数
    def predict(self, text, max_len=32, end_word=None):
        # 终止标志
        if end_word is not None:
            end_id = self.tokenizer.encode(end_word)
            length = len(end_id)
        else:
            end_id = self.tokenizer.eod_id

        # 初始预测
        ids = self.tokenizer.encode(text)
        input_id = paddle.to_tensor(np.array(ids).reshape(1, -1).astype('int64'))
        output, cached_kvs = self.model(input_id, use_cache=True)
        nid = int(np.argmax(output[0, -1].numpy()))
        out = [nid]

        # 使用缓存进行继续预测
        for i in range(max_len-1):
            input_id = paddle.to_tensor(np.array([nid]).reshape(1, -1).astype('int64'))
            output, cached_kvs = self.model(input_id, cached_kvs, use_cache=True)
            nid = int(np.argmax(output[0, -1].numpy()))

            # 根据终止标志停止预测
            if (end_word is not None) and (out[-length+1:]+[nid]==end_id):
                break
            elif (end_word is None) and (nid==end_id):
                break
            
            out.append(nid)
        
        return self.tokenizer.decode(out)