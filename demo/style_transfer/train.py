import paddle
import paddlehub as hub
import paddlehub.transforms.transforms as T
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets import MiniCOCO

if __name__ == "__main__":
    model = hub.Module(name='msgnet')
    transform = T.Compose([T.Resize(
        (256, 256), interp='LINEAR'), T.CenterCrop(crop_size=256)], T.SetType(datatype='float32'))

    styledata = MiniCOCO(transform)
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='img_style_transfer_ckpt')
    trainer.train(styledata, epochs=5, batch_size=16, eval_dataset=styledata, log_interval=1, save_interval=1)
