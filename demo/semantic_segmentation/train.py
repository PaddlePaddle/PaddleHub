import paddle
import paddlehub as hub
from paddlehub.finetune.trainer import Trainer

from paddlehub.datasets import OpticDiscSeg
from paddlehub.vision.segmentation_transforms import Compose, Resize, Normalize

if __name__ == "__main__":
    transform = Compose([Resize(target_size=(512, 512)), Normalize()])
    train_reader = OpticDiscSeg(transform)

    model = hub.Module(name='ocrnet_hrnetw18_voc', num_classes=2)
    scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=1000, power=0.9,  end_lr=0.0001)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_ocr', use_gpu=True)
    trainer.train(train_reader, epochs=20, batch_size=4, eval_dataset=train_reader, log_interval=10, save_interval=4)