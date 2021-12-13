import os
import paddle
import numpy as np
import paddle.nn as nn
from paddlehub.module.module import moduleinfo, serving
from paddlenlp.transformers import GPT2ForPretraining, GPT2ChineseTokenizer, GPT2Model


@moduleinfo(
    name="GPT2_CPM_LM",  # 模型名称
    type="NLP/NLG",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="GPT2_CPM_LM",  # 模型介绍
    version="1.0.0"  # 版本号
)
class GPT2_CPM_LM(nn.Layer):
    def __init__(self):
        super(GPT2_CPM_LM, self).__init__()
        # 实例化模型
        gpt2 = GPT2Model(
            vocab_size=30000,
            hidden_size=2560,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=10240,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=1024,
            type_vocab_size=1,
            initializer_range=0.02,
            pad_token_id=0)
        self.model = GPT2ForPretraining(gpt2)

        # 读取CPM-LM模型参数(FP16)
        state_dict = paddle.load(os.path.join(self.directory, 'CPM-LM.pdparams'))

        # FP16 -> FP32
        for param in state_dict:
            state_dict[param] = state_dict[param].astype('float32')

        # 设置模型参数
        self.model.set_dict(state_dict)

        # 将模型设置为评估状态
        self.model.eval()

        # 加载编解码器
        self.tokenizer = GPT2ChineseTokenizer(
            vocab_file=os.path.join(self.directory, 'vocab.json'),
            model_file=os.path.join(self.directory, 'chinese_vocab.model'))

        # 初始化编码器
        _ = self.tokenizer.encode('_')

    # Greedy Search
    def greedy_search(self, text, max_len=32, end_word=None):
        with paddle.no_grad():
            # # 终止标志
            if end_word is not None:
                stop_id = self.tokenizer.encode(end_word)
                length = len(stop_id)

            # 初始预测
            ids = self.tokenizer.encode(text)
            input_id = paddle.to_tensor(np.array(ids).reshape(1, -1).astype('int64'))
            output, cached_kvs = self.model(input_id, use_cache=True)
            next_token = int(np.argmax(output[0, -1].numpy()))
            ids.append(next_token)

            # 使用缓存进行继续预测
            for i in range(max_len - 1):
                input_id = paddle.to_tensor(np.array([next_token]).reshape(1, -1).astype('int64'))
                output, cached_kvs = self.model(input_id, use_cache=True, cache=cached_kvs)
                next_token = int(np.argmax(output[0, -1].numpy()))
                ids.append(next_token)

                # 根据终止标志停止预测
                if (end_word is not None) and (ids[-length:] == stop_id):
                    break

            return self.tokenizer.decode(ids)

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(top_k, logits.shape[-1])  # Safety check
        logits_np = logits.numpy()
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits_np < np.sort(logits_np)[-top_k]
            logits_np[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits = paddle.sort(logits, descending=True)
            sorted_indices = paddle.argsort(logits, descending=True).numpy()
            cumulative_probs = paddle.cumsum(paddle.nn.functional.softmax(sorted_logits, axis=-1), axis=-1).numpy()

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits_np[indices_to_remove] = filter_value

        return paddle.to_tensor(logits_np)

    def nucleus_sample(self,
                       text,
                       max_len=32,
                       end_word=None,
                       repitition_penalty=1.0,
                       temperature=1.0,
                       top_k=0,
                       top_p=1.0):
        with paddle.no_grad():
            # 终止标志
            if end_word is not None:
                stop_id = self.tokenizer.encode(end_word)
                length = len(stop_id)

            # 初始预测
            ids = self.tokenizer.encode(text)
            input_id = paddle.to_tensor(np.array(ids).reshape(1, -1).astype('int64'))
            output, cached_kvs = self.model(input_id, use_cache=True)
            next_token_logits = output[0, -1, :]
            for id in set(ids):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[self.tokenizer.encoder['<unk>']] = -float('Inf')
            filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = paddle.multinomial(
                paddle.nn.functional.softmax(filtered_logits, axis=-1), num_samples=1).numpy()
            ids += [int(next_token)]

            # 使用缓存进行继续预测
            for i in range(max_len - 1):
                input_id = paddle.to_tensor(np.array([next_token]).reshape(1, -1).astype('int64'))
                output, cached_kvs = self.model(input_id, use_cache=True, cache=cached_kvs)
                next_token_logits = output[0, -1, :]
                for id in set(ids):
                    next_token_logits[id] /= repitition_penalty
                next_token_logits = next_token_logits / temperature
                next_token_logits[self.tokenizer.encoder['<unk>']] = -float('Inf')
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = paddle.multinomial(
                    paddle.nn.functional.softmax(filtered_logits, axis=-1), num_samples=1).numpy()
                ids += [int(next_token)]

                # 根据终止标志停止预测
                if (end_word is not None) and (ids[-length:] == stop_id):
                    break

            return self.tokenizer.decode(ids)

    # Hub Serving
    @serving
    def serving_method(self, text, mode='search', **kwargs):
        if mode == 'search':
            results = self.greedy_search(text, **kwargs)
        else:
            results = self.nucleus_sample(text, **kwargs)

        return results
