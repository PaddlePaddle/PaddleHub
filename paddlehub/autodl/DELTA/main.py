

import os
import time
import sys
import math
import numpy as np
import functools
import re
import logging
import glob

import paddle
import paddle.fluid as fluid
from models.resnet import ResNet101
from datasets.readers import ReaderConfig

# import cv2
# import skimage
# import matplotlib.pyplot as plt
# from paddle.fluid.core import PaddleTensor
# from paddle.fluid.core import AnalysisConfig
# from paddle.fluid.core import create_paddle_predictor

from args import args
from datasets.data_path import global_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if args.seed is not None:
    np.random.seed(args.seed)

print(os.environ.get('LD_LIBRARY_PATH', None))
print(os.environ.get('PATH', None))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_vars_by_dict(executor, name_var_dict, main_program=None):
    from paddle.fluid.framework import Program, Variable
    from paddle.fluid import core

    load_prog = Program()
    load_block = load_prog.global_block()

    if main_program is None:
        main_program = fluid.default_main_program()

    if not isinstance(main_program, Program):
        raise TypeError("program should be as Program type or None")

    for each_var_name in name_var_dict.keys():
        assert isinstance(name_var_dict[each_var_name], Variable)
        if name_var_dict[each_var_name].type == core.VarDesc.VarType.RAW:
            continue

        load_block.append_op(
            type='load',
            inputs={},
            outputs={'Out': [name_var_dict[each_var_name]]},
            attrs={'file_path': each_var_name}
        )

    executor.run(load_prog)


def get_model_id():
    prefix = ''
    if args.prefix is not None:
        prefix = args.prefix + '-'  # for some notes.

    model_id = prefix + args.dataset + \
               '-epo_' + str(args.num_epoch) + \
               '-b_' + str(args.batch_size) + \
               '-reg_' + str(args.delta_reg) + \
               '-wd_' + str(args.wd_rate)
    return model_id


def train():
    dataset = args.dataset
    image_shape = [3, 224, 224]
    pretrained_model = args.pretrained_model

    class_map_path = f'{global_data_path}/{dataset}/readable_label.txt'

    if os.path.exists(class_map_path):
        logger.info("The map of readable label and numerical label has been found!")
        with open(class_map_path) as f:
            label_dict = {}
            strinfo = re.compile(r"\d+ ")
            for item in f.readlines():
                key = int(item.split(" ")[0])
                value = [
                    strinfo.sub("", l).replace("\n", "")
                    for l in item.split(", ")
                ]
                label_dict[key] = value[0]

    assert os.path.isdir(pretrained_model), "please load right pretrained model path for infer"

    # data reader
    batch_size = args.batch_size
    reader_config = ReaderConfig(f'{global_data_path}/{dataset}', is_test=False)
    reader = reader_config.get_reader()
    train_reader = paddle.batch(
        paddle.reader.shuffle(reader, buf_size=batch_size),
        batch_size,
        drop_last=True)

    # model ops
    image = fluid.data(name='image', shape=[None] + image_shape, dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    model = ResNet101(is_test=False)
    features, logits = model.net(input=image, class_dim=reader_config.num_classes)
    out = fluid.layers.softmax(logits)

    # loss, metric
    cost = fluid.layers.mean(fluid.layers.cross_entropy(out, label))
    accuracy = fluid.layers.accuracy(input=out, label=label)

    # delta regularization
    # teacher model pre-trained on Imagenet, 1000 classes.
    global_name = 't_'
    t_model = ResNet101(is_test=True, global_name=global_name)
    t_features, _ = t_model.net(input=image, class_dim=1000)
    for f in t_features.keys():
        t_features[f].stop_gradient = True

    # delta loss. hard code for the layer name, which is just before global pooling.
    delta_loss = fluid.layers.square(t_features['t_res5c.add.output.5.tmp_0'] - features['res5c.add.output.5.tmp_0'])
    delta_loss = fluid.layers.reduce_mean(delta_loss)

    params = fluid.default_main_program().global_block().all_parameters()
    parameters = []
    for param in params:
        if param.trainable:
            if global_name in param.name:
                print('\tfixing', param.name)
            else:
                print('\ttraining', param.name)
                parameters.append(param.name)

    # optimizer, with piecewise_decay learning rate.
    total_steps = len(reader_config.image_paths) * args.num_epoch // batch_size
    boundaries = [int(total_steps * 2 / 3)]
    print('\ttotal learning steps:', total_steps)
    print('\tlr decays at:', boundaries)
    values = [0.01, 0.001]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
        momentum=0.9, parameter_list=parameters,
        regularization=fluid.regularizer.L2Decay(args.wd_rate)
    )
    cur_lr = optimizer._global_learning_rate()

    optimizer.minimize(cost + args.delta_reg * delta_loss,
                       parameter_list=parameters)

    # data reader
    feed_order = ['image', 'label']

    # executor (session)
    place = fluid.CUDAPlace(args.use_cuda) if args.use_cuda >= 0 else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # running
    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()

    feed_var_list_loop = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
    exe.run(start_program)

    loading_parameters = {}
    t_loading_parameters = {}
    for p in main_program.all_parameters():
        if 'fc' not in p.name:
            if global_name in p.name:
                new_name = os.path.join(pretrained_model, p.name.split(global_name)[-1])
                t_loading_parameters[new_name] = p
                print(new_name, p.name)
            else:
                name = os.path.join(pretrained_model, p.name)
                loading_parameters[name] = p
                print(name, p.name)
        else:
            print(f'not loading {p.name}')

    load_vars_by_dict(exe, loading_parameters, main_program=main_program)
    load_vars_by_dict(exe, t_loading_parameters, main_program=main_program)

    step = 0

    # test_data = reader_creator_all_in_memory('./datasets/PetImages', is_test=True)
    for e_id in range(args.num_epoch):
        avg_delta_loss = AverageMeter()
        avg_loss = AverageMeter()
        avg_accuracy = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()

        for step_id, data_train in enumerate(train_reader()):
            wrapped_results = exe.run(
                main_program,
                feed=feeder.feed(data_train),
                fetch_list=[cost, accuracy, delta_loss, cur_lr])
            # print(avg_loss_value[2])
            batch_time.update(time.time() - end)
            end = time.time()

            avg_loss.update(wrapped_results[0][0], len(data_train))
            avg_accuracy.update(wrapped_results[1][0], len(data_train))
            avg_delta_loss.update(wrapped_results[2][0], len(data_train))
            if step % 100 == 0:
                print(f"\tEpoch {e_id}, Global_Step {step}, Batch_Time {batch_time.avg: .2f},"
                      f" LR {wrapped_results[3][0]}, "
                      f"Loss {avg_loss.avg: .4f}, Acc {avg_accuracy.avg: .4f}, Delta_Loss {avg_delta_loss.avg: .4f}"
                      )
            step += 1

        if args.outdir is not None:
            try:
                os.makedirs(args.outdir, exist_ok=True)
                fluid.io.save_params(executor=exe, dirname=args.outdir + '/' + get_model_id())
            except:
                print('\t Not saving trained parameters.')

        if e_id == args.num_epoch - 1:
            print("kpis\ttrain_cost\t%f" % avg_loss.avg)
            print("kpis\ttrain_acc\t%f" % avg_accuracy.avg)


def test():
    image_shape = [3, 224, 224]
    pretrained_model = args.outdir + '/' + get_model_id()

    # data reader
    batch_size = args.batch_size
    reader_config = ReaderConfig(f'{global_data_path}/{args.dataset}', is_test=True)
    reader = reader_config.get_reader()
    test_reader = paddle.batch(reader, batch_size)

    # model ops
    image = fluid.data(name='image', shape=[None] + image_shape, dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    model = ResNet101(is_test=True)
    _, logits = model.net(input=image, class_dim=reader_config.num_classes)
    out = fluid.layers.softmax(logits)

    # loss, metric
    cost = fluid.layers.mean(fluid.layers.cross_entropy(out, label))
    accuracy = fluid.layers.accuracy(input=out, label=label)

    # data reader
    feed_order = ['image', 'label']

    # executor (session)
    place = fluid.CUDAPlace(args.use_cuda) if args.use_cuda >= 0 else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # running
    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()

    feed_var_list_loop = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
    exe.run(start_program)

    fluid.io.load_params(exe, pretrained_model)

    step = 0
    avg_loss = AverageMeter()
    avg_accuracy = AverageMeter()

    for step_id, data_train in enumerate(test_reader()):
        avg_loss_value = exe.run(
            main_program,
            feed=feeder.feed(data_train),
            fetch_list=[cost, accuracy])
        avg_loss.update(avg_loss_value[0], len(data_train))
        avg_accuracy.update(avg_loss_value[1], len(data_train))
        if step_id % 10 == 0:
            print("\nBatch %d, Loss %f, Acc %f" % (step_id, avg_loss.avg, avg_accuracy.avg))
        step += 1

    print("test counts:", avg_loss.count)
    print("test_cost\t%f" % avg_loss.avg)
    print("test_acc\t%f" % avg_accuracy.avg)


if __name__ == '__main__':
    print(args)
    train()
    test()

# srun python -u training.py --prefix 'v1' --dataset benchmark/Caltech30 --delta_reg 0.0 --wd_rate 1e-4 --batch_size 64 --outdir outdir --num_epoch 100 --use_cuda 0
# srun python -u training.py --prefix 'v1' --dataset benchmark/Caltech30 --delta_reg 0.1 --wd_rate 1e-4 --batch_size 64 --outdir outdir --num_epoch 100 --use_cuda 0
# srun python -u training.py --prefix 'v1' --dataset benchmark/CUB_200_2011 --delta_reg 0.0 --wd_rate 1e-4 --batch_size 64 --outdir outdir --num_epoch 100 --use_cuda 0
# srun python -u training.py --prefix 'v1' --dataset benchmark/CUB_200_2011 --delta_reg 0.1 --wd_rate 1e-4 --batch_size 64 --outdir outdir --num_epoch 100 --use_cuda 0