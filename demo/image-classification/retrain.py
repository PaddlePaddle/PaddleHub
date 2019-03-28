import paddle_hub as hub
import paddle
import paddle.fluid as fluid
from paddle_hub.dataset.flowers import FlowersDataset
from paddle_hub.dataset.dogcat import DogCatDataset
from paddle_hub.dataset.cv_reader import ImageClassificationReader
from paddle_hub.finetune.task import Task
from paddle_hub.finetune.network import append_mlp_classifier
from paddle_hub.finetune.config import FinetuneConfig
from paddle_hub.finetune.finetune import finetune_and_eval


def train():
    resnet_module = hub.Module(module_dir="./hub_module_ResNet50")
    input_dict, output_dict, program = resnet_module.context(
        sign_name="feature_map")
    data_processor = ImageClassificationReader(
        image_width=224,
        image_height=224,
        dataset=FlowersDataset(),
        color_mode="RGB")
    with fluid.program_guard(program):
        label = fluid.layers.data(name="label", dtype="int64", shape=[1])
        img = input_dict[0]
        feature_map = output_dict[0]

        config = FinetuneConfig(
            log_interval=10,
            eval_interval=100,
            use_cuda=True,
            learning_rate=1e-4,
            weight_decay=None,
            in_tokens=None,
            num_epoch=10,
            batch_size=32,
            max_seq_len=None,
            warmup_proportion=None,
            save_ckpt_interval=200,
            checkpoint_dir="./finetune_task",
            strategy='BaseFinetune',
            with_memory_optimization=True)

        feed_list = [img.name, label.name]

        task = append_mlp_classifier(
            feature=feature_map, label=label, num_classes=5)
        finetune_and_eval(
            task,
            feed_list=feed_list,
            data_processor=data_processor,
            config=config)


if __name__ == "__main__":
    train()
