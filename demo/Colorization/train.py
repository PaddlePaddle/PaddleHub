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
        transform = Compose([Resize((256,256),interp="RANDOM"),RandomPaddingCrop(crop_size=176), ColorConvert(mode='RGB2LAB'), ColorizePreprocess(ab_thresh=0, p=1)])
        color_set = Colorization(transform=transform, mode=is_train)
        if is_train:
            model.train()
            optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.0001, parameter_list=model.parameters())
            trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
            trainer.train(color_set, epochs=3, batch_size=1, eval_dataset=color_set, save_interval=1)

