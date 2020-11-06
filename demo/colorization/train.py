import paddle
import paddlehub as hub
import paddlehub.vision.transforms as T
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets import Canvas

if __name__ == '__main__':

    model = hub.Module(name='user_guided_colorization', classification=True, prob= 0.125)
    transform = T.Compose([T.Resize((256, 256), interpolation='NEAREST'),
                           T.RandomPaddingCrop(crop_size=176),
                           T.RGB2LAB()])

    color_set = Canvas(transform=transform, mode='train')
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='img_colorization_ckpt')
    trainer.train(color_set, epochs=101, batch_size=2, eval_dataset=color_set, log_interval=10, save_interval=10)
