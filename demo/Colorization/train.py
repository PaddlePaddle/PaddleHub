from paddlehub.finetune.trainer import Trainer
import paddle.fluid as fluid
import paddlehub as hub
import paddle
import paddle.nn as nn

from hub_module.modules.image.Colorization.User_guided_colorization.module import Userguidedcolorization
from paddlehub.datasets.colorizedataset import  Colorizedataset
from paddlehub.process.transforms import import *


if __name__ == '__main__':
    is_train = True
    paddle.disable_static()
    model = Userguidedcolorization()
    transform = Compose([Resize((256,256),interp="RANDOM"),RandomPaddingCrop(crop_size=176), ConvertColorSpace(mode='RGB2LAB'), ColorizePreprocess(ab_thresh=0, p=1)], stay_rgb=True)
    color_set = Colorizedataset(transform=transform, mode=is_train)
    if is_train:
        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.0001, parameter_list=model.parameters())
        trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
        trainer.train(color_set, epochs=3, batch_size=1, eval_dataset=color_set, save_interval=1)


