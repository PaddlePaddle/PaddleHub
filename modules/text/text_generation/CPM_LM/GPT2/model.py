import math
import paddle
import paddle.nn as nn

class MLP(nn.Layer):
    def __init__(self, embedding_size):
        super(MLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(embedding_size, embedding_size*4)
        self.dense_4h_to_h = nn.Linear(embedding_size*4, embedding_size)
        self.act = nn.functional.gelu

    def forward(self, x):
        h = self.act(self.dense_h_to_4h(x))
        h2 = self.dense_4h_to_h(h)
        return h2

class Attention(nn.Layer):
    def __init__(self, 
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = embedding_size // num_attention_heads
        self.embedding_size = embedding_size

        self.query_key_value = nn.Linear(embedding_size, embedding_size * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.resid_drop = nn.Dropout(residual_dropout)
        self.dense = nn.Linear(embedding_size, embedding_size)

    def split_heads(self, x):
        x = x.reshape([-1, self.seq_len, self.num_attention_heads, self.size_per_head])
        return x.transpose((0, 2, 1, 3))

    def forward(self, x, kv_cache=None):
        self.seq_len = x.shape[1]
        x = self.query_key_value(x)
        q, k, v = x.split(num_or_sections=3, axis=2)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        if kv_cache is not None:
            pk, pv = paddle.unstack(kv_cache, axis=1)
            k = paddle.concat([pk, k], axis=-2)
            v = paddle.concat([pv, v], axis=-2)
        cached_kv = paddle.stack([k, v], axis=1)

        attn = paddle.matmul(q, k, transpose_y=True)  # [B, N, L, S]
        attn = attn / math.sqrt(self.size_per_head)

        # [L, S]
        attention_mask = paddle.tril(paddle.ones([self.seq_len, self.seq_len], 'float32'))
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.seq_len])

        # adding to softmax -> its like removing them entirely
        attn = attn * attention_mask - 10000.0 * (1.0 - attention_mask)
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        y = paddle.matmul(attn, v)
        # [B, N, L, S] -> [B, L, N, S]
        y = y.transpose((0, 2, 1, 3))
        y = paddle.reshape(y, [-1, self.seq_len, self.embedding_size])
        y = self.resid_drop(self.dense(y))

        return y, cached_kv

class Block(nn.Layer):
    def __init__(self, 
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout):
        super(Block, self).__init__()
        self.input_layernorm = nn.LayerNorm(embedding_size, epsilon=1e-5)
        self.attention = Attention(embedding_size, num_attention_heads, attention_dropout, residual_dropout)
        self.post_attention_layernorm = nn.LayerNorm(embedding_size, epsilon=1e-5)
        self.mlp = MLP(embedding_size)

    def forward(self, x, kv_cache=None):
        attn, cached_kv = self.attention(self.input_layernorm(x), kv_cache=kv_cache)
        x = x + attn
        z = self.post_attention_layernorm(x)
        z = self.mlp(z)
        x = x + z
        return x, cached_kv

class Transformer(nn.Layer):
    def __init__(self, 
                layer_size,
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout):
        super(Transformer, self).__init__()

        self.layers = nn.LayerList([Block(
                embedding_size, 
                num_attention_heads,
                attention_dropout,
                residual_dropout) 
            for _ in range(layer_size)])

        self.final_layernorm = nn.LayerNorm(embedding_size, epsilon=1e-5)
    
    def forward(self, x, kv_cache=None):
        cached_kvs = []
        for i, layer in enumerate(self.layers):
            x, cached_kv = layer(x, kv_cache=kv_cache[i] if kv_cache is not None else None)
            cached_kvs.append(cached_kv)
        x = self.final_layernorm(x)
        return x, paddle.stack(cached_kvs)

class GPT2Model(nn.Layer):
    def __init__(self,
                 vocab_size,
                 layer_size,
                 block_size,
                 embedding_dropout,
                 embedding_size,
                 num_attention_heads,
                 attention_dropout,
                 residual_dropout):
        super(GPT2Model, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(block_size, embedding_size)
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.transformer = Transformer(
            layer_size,
            embedding_size, 
            num_attention_heads,
            attention_dropout,
            residual_dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        if kv_cache is None:
            past_length = 0
        else:
            past_length = kv_cache[0][0].shape[-2]

        position_ids = paddle.arange(past_length, x.shape[-1] + past_length, dtype='int64')
        position_ids = position_ids.unsqueeze(0).expand_as(x)

        x = self.word_embeddings(x)
        x = self.emb_drop(x + self.position_embeddings(position_ids))
        x, cached_kvs = self.transformer(x, kv_cache)
        x = paddle.matmul(x, self.word_embeddings.weight, transpose_y=True)

        if use_cache:
            return x, cached_kvs
        else:
            return x

if __name__ == '__main__':
    gpt = GPT2Model(
    vocab_size=30000,
    layer_size=2,
    block_size=1024,
    embedding_dropout=0.0,
    embedding_size=2560,
    num_attention_heads=32,
    attention_dropout=0.0,
    residual_dropout=0.0)
    gpt.eval()
    out, cached_kvs = gpt(paddle.ones([1,1], 'int64'), paddle.randn([32, 1, 2, 32, 9, 80], 'float32'), use_cache=True)
    print(out.shape, cached_kvs.shape)
