import paddle
import paddle.nn as nn
from paddlenlp.transformers.clip.modeling import TextTransformer


class CLIPText(nn.Layer):
    def __init__(
            self,
            max_text_length: int = 77,
            vocab_size: int = 49408,
            text_embed_dim: int = 512,
            text_heads: int = 8,
            text_layers: int = 12,
            text_hidden_act: str = "quick_gelu",
            projection_dim: int = 512):
        super().__init__()

        self.text_model = TextTransformer(context_length=max_text_length,
                                          transformer_width=text_embed_dim,
                                          transformer_heads=text_heads,
                                          transformer_layers=text_layers,
                                          vocab_size=vocab_size,
                                          activation=text_hidden_act,
                                          normalize_before=True)

        self.text_projection = paddle.create_parameter(
            (text_embed_dim, projection_dim), paddle.get_default_dtype())

    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        pooled_output = text_outputs[1]
        text_features = paddle.matmul(pooled_output, self.text_projection)
        return text_features
