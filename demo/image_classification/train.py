import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddlehub.finetune.trainer import Trainer
from paddlehub.datasets.flowers import Flowers
from paddlehub.process.transforms import Compose, Resize, Normalize
from paddlehub.module.cv_module import ImageClassifierModule

if __name__ == '__main__':
    with fluid.dygraph.guard(fluid.CUDAPlace(ParallelEnv().dev_id)):
        transforms = Compose([Resize((224, 224)), Normalize()])
        flowers = Flowers(transforms)
        flowers_validate = Flowers(transforms, mode='val')

        model = hub.Module(directory='mobilenet_v2_animals', class_dim=flowers.num_classes)
        # model = hub.Module(name='mobilenet_v2_animals', class_dim=flowers.num_classes)

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001, parameter_list=model.parameters())
        trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_cls')

        trainer.train(flowers, epochs=100, batch_size=32, eval_dataset=flowers_validate, save_interval=1)
