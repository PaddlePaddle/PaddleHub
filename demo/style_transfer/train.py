import paddle
import paddlehub as hub

from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets.minicoco import MiniCOCO
from paddlehub.vision.transforms import Compose, Resize, CenterCrop

if __name__ == "__main__":
    model = hub.Module(name='msgnet')
    transform = Compose([Resize((256, 256), interpolation='LINEAR')])
    styledata = MiniCOCO(transform)
    scheduler =  paddle.optimizer.lr.PolynomialDecay(learning_rate=0.001, power=0.9, decay_steps=100)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img')
    trainer.train(styledata, epochs=101, batch_size=4, eval_dataset=styledata, log_interval=10, save_interval=10)
