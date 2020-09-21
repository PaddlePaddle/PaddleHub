import paddle
import paddlehub as hub
import paddle.nn as nn
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets.pascalvoc import DetectionData
from paddlehub.process.transforms import DetectTrainReader, DetectTestReader

if __name__ == "__main__":
    place = paddle.CUDAPlace(0)
    paddle.disable_static()
    is_train = True
    if is_train:
        transform = DetectTrainReader()
        train_reader = DetectionData(transform)
    else:
        transform = DetectTestReader()
        test_reader = DetectionData(transform)
    model = hub.Module(name='yolov3_darknet53_pascalvoc')
    model.train()
    optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')
    trainer.train(train_reader, epochs=5, batch_size=4, eval_dataset=train_reader, log_interval=1, save_interval=1)
