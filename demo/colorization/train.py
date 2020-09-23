import paddle
import paddlehub as hub
import paddle.nn as nn
from paddlehub.finetune.trainer import Trainer

from paddlehub.datasets.colorizedataset import Colorizedataset
from paddlehub.process.transforms import Compose, Resize, RandomPaddingCrop, ConvertColorSpace, ColorizePreprocess

if __name__ == '__main__':
    is_train = True
    paddle.disable_static()
    model = hub.Module(name='user_guided_colorization')
    transform = Compose([
        Resize((256, 256), interp='NEAREST'),
        RandomPaddingCrop(crop_size=176),
        ConvertColorSpace(mode='RGB2LAB'),
        ColorizePreprocess(ab_thresh=0, is_train=is_train),
    ],
                        stay_rgb=True,
                        is_permute=False)
    color_set = Colorizedataset(transform=transform, mode='train')
    if is_train:
        model.train()
        optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
        trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
        trainer.train(color_set, epochs=101, batch_size=5, eval_dataset=color_set, log_interval=10, save_interval=10)
