import paddle
import paddlehub as hub

from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets.styletransfer import StyleTransferData
from paddlehub.process.transforms import Compose, Resize, CenterCrop, SetType

if __name__ == "__main__":
    place = paddle.CUDAPlace(0)
    paddle.disable_static()
    model = hub.Module(name='msgnet')
    transform = Compose([Resize((256, 256), interp='LINEAR'), CenterCrop(crop_size=256)], SetType(datatype='float32'))
    styledata = StyleTransferData(transform)
    model.train()
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
    trainer.train(styledata, epochs=5, batch_size=1, eval_dataset=styledata, log_interval=1, save_interval=1)
