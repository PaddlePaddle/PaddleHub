from paddlehub.finetune.trainer import Trainer
import paddle.fluid as fluid
import paddlehub as hub
import paddle
import paddle.nn as nn
from module import Userguidedcolorization

from dataset import Colorization

from process.transforms import *
if __name__ == '__main__':
    place = fluid.CPUPlace()
    is_train = True

    with fluid.dygraph.guard(place):
        model = Userguidedcolorization()
        model.eval()
        result = model.predict(images='/Users/haoyuying/Downloads/超分数据集/test/SRF_2/data/Urban100_097.png')

