import paddle.nn as nn

from .vit import ViT
from .clip import CLIPText
from .scratch import Scratch


class LSeg(nn.Layer):
    def __init__(self):
        super().__init__()
        self.clip = CLIPText()
        self.vit = ViT()
        self.scratch = Scratch()

    def forward(self, images, texts):
        layer_1, layer_2, layer_3, layer_4 = self.vit.forward(images)
        text_features = self.clip.get_text_features(texts)
        return self.scratch.forward(layer_1, layer_2, layer_3, layer_4, text_features)
