import paddle
import paddlehub as hub
import paddle.nn as nn
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets.pascalvoc import DetectionData
import paddlehub.process.detect_transforms as T
if __name__ == "__main__":
    place = paddle.CUDAPlace(0)
    paddle.disable_static()

    transform = T.Compose([
        T.RandomDistort(),
        T.RandomExpand(fill=[0.485, 0.456, 0.406]),
        T.RandomCrop(),
        T.Resize(target_size=416),
        T.RandomFlip(),
        T.ShuffleBox(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_reader = DetectionData(transform)
    model = hub.Module(name='yolov3_darknet53_pascalvoc')
    model.train()
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
    trainer.train(train_reader, epochs=5, batch_size=4, eval_dataset=train_reader, log_interval=1, save_interval=1)
