import paddle
import paddlehub as hub

from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets.minicoco import MiniCOCO
import paddlehub.vision.transforms as T

if __name__ == "__main__":
    model = hub.Module(name='msgnet')
    transform = T.Compose([T.Resize((256, 256), interpolation='LINEAR')])
    styledata = MiniCOCO(transform)
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='test_style_ckpt')
    trainer.train(styledata, epochs=101, batch_size=4, log_interval=10, save_interval=10)
