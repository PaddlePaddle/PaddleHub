import paddle
import paddlehub as hub
import paddle.nn as nn
from paddlehub.finetune.trainer import Trainer

from paddlehub.datasets.colorizedataset import  Colorizedataset
from paddlehub.process.transforms import Compose, Resize, RandomPaddingCrop, ConvertColorSpace,  ColorizePreprocess


if __name__ == '__main__':
    is_train = True
    paddle.disable_static()
    model = hub.Module(directory='user_guided_colorization')
    transform = Compose([Resize((256,256),interp="RANDOM"),RandomPaddingCrop(crop_size=176), ConvertColorSpace(mode='RGB2LAB'), ColorizePreprocess(ab_thresh=0, p=1)], stay_rgb=True)
    color_set = Colorizedataset(transform=transform, mode=is_train)
    if is_train:
        model.train()
        optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
        trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
        trainer.train(color_set, epochs=3, batch_size=1, eval_dataset=color_set, save_interval=1)
