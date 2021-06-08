import paddle
import paddlehub as hub
import paddlehub.vision.transforms as T
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets import Canvas

if __name__ == '__main__':

    transform = T.Compose(
        [T.Resize((256, 256), interpolation='NEAREST'),
         T.RandomPaddingCrop(crop_size=176),
         T.RGB2LAB()], to_rgb=True)

    color_set = Canvas(transform=transform, mode='train')
    model = hub.Module(name='user_guided_colorization', load_checkpoint='/PATH/TO/CHECKPOINT')

    model.set_config(classification=True, prob=1)
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='img_colorization_ckpt_cls_1')
    trainer.train(color_set, epochs=201, batch_size=25, eval_dataset=color_set, log_interval=10, save_interval=10)

    model.set_config(classification=False, prob=0.125)
    optimizer = paddle.optimizer.Adam(learning_rate=0.00001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='img_colorization_ckpt_reg_1')
    trainer.train(color_set, epochs=101, batch_size=25, log_interval=10, save_interval=10)
