#coding:utf-8
import argparse
import os

import numpy as np
import paddlehub as hub
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.optimizer import AdamOptimizer

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch",          type=int,               default=1,                                  help="Number of epoches for fine-tuning.")
parser.add_argument("--checkpoint_dir",     type=str,               default="paddlehub_finetune_ckpt_dygraph",  help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=16,                                 help="Total examples' number in batch for training.")
parser.add_argument("--log_interval",       type=int,               default=10,                                 help="log interval.")
parser.add_argument("--save_interval",      type=int,               default=10,                                 help="save interval.")
# yapf: enable.


class ResNet50(fluid.dygraph.Layer):
    def __init__(self, num_classes, backbone):
        super(ResNet50, self).__init__()
        self.fc = Linear(input_dim=2048, output_dim=num_classes)
        self.backbone = backbone

    def forward(self, imgs):
        feature_map = self.backbone(imgs)
        feature_map = fluid.layers.reshape(feature_map, shape=[-1, 2048])
        pred = self.fc(feature_map)
        return fluid.layers.softmax(pred)


def finetune(args):
    with fluid.dygraph.guard():
        resnet50_vd_10w = hub.Module(name="resnet50_vd_10w")
        dataset = hub.dataset.Flowers()
        resnet = ResNet50(
            num_classes=dataset.num_labels, backbone=resnet50_vd_10w)
        adam = AdamOptimizer(
            learning_rate=0.001, parameter_list=resnet.parameters())
        state_dict_path = os.path.join(args.checkpoint_dir,
                                       'dygraph_state_dict')
        if os.path.exists(state_dict_path + '.pdparams'):
            state_dict, _ = fluid.load_dygraph(state_dict_path)
            resnet.load_dict(state_dict)

        reader = hub.reader.ImageClassificationReader(
            image_width=resnet50_vd_10w.get_expected_image_width(),
            image_height=resnet50_vd_10w.get_expected_image_height(),
            images_mean=resnet50_vd_10w.get_pretrained_images_mean(),
            images_std=resnet50_vd_10w.get_pretrained_images_std(),
            dataset=dataset)
        train_reader = reader.data_generator(
            batch_size=args.batch_size, phase='train')

        loss_sum = acc_sum = cnt = 0
        # 执行epoch_num次训练
        for epoch in range(args.num_epoch):
            # 读取训练数据进行训练
            for batch_id, data in enumerate(train_reader()):
                imgs = np.array(data[0][0])
                labels = np.array(data[0][1])

                pred = resnet(imgs)
                acc = fluid.layers.accuracy(pred, to_variable(labels))
                loss = fluid.layers.cross_entropy(pred, to_variable(labels))
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                # 参数更新
                adam.minimize(avg_loss)

                loss_sum += avg_loss.numpy() * imgs.shape[0]
                acc_sum += acc.numpy() * imgs.shape[0]
                cnt += imgs.shape[0]
                if batch_id % args.log_interval == 0:
                    print('epoch {}: loss {}, acc {}'.format(
                        epoch, loss_sum / cnt, acc_sum / cnt))
                    loss_sum = acc_sum = cnt = 0

                if batch_id % args.save_interval == 0:
                    state_dict = resnet.state_dict()
                    fluid.save_dygraph(state_dict, state_dict_path)


if __name__ == "__main__":
    args = parser.parse_args()
    finetune(args)
