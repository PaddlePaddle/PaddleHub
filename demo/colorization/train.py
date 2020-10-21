import paddle
import paddlehub as hub
import paddle.nn as nn
from paddlehub.finetune.trainer import Trainer

from paddlehub.datasets.colorizedataset import Colorizedataset
import paddlehub.process.transforms as T

if __name__ == '__main__':

    paddle.disable_static()
    model = hub.Module(name='user_guided_colorization')
    transform = T.Compose([T.Resize((256, 256), interp='NEAREST'),
                           T.RandomPaddingCrop(crop_size=176),
                           T.RGB2LAB()],
                          stay_rgb=True,
                          is_permute=False)
    color_set = Colorizedataset(transform=transform, mode='train')
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
    trainer.train(color_set, epochs=101, batch_size=2, eval_dataset=color_set, log_interval=10, save_interval=10)
