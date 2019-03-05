from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
import paddle.dataset.flowers as flowers
import reader
import argparse
import functools
import subprocess
import utils
import nets
import paddle_hub as hub
from utils.learning_rate import cosine_decay
from utils.fp16_utils import create_master_params_grads, master_param_to_train_param
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('create_module',    bool, False,                  "create a hub module or not" )
add_arg('batch_size',       int,   32,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   12000,              "Training image number.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   2,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('model_save_dir',   str,   "output",             "model save directory")
add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "ResNet50", "Set the network to use.")
add_arg('data_dir',         str,   "./dataset",  "The ImageNet dataset root dir.")
add_arg('fp16',             bool,  False,                "Enable half precision training with fp16." )
add_arg('scale_loss',       float, 1.0,                  "Scale loss for fp16." )
# yapf: enable


def optimizer_setting(params):
    ls = params["learning_strategy"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 12000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = 12000
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(4e-5))
    elif ls["name"] == "exponential_decay":
        if "total_images" not in params:
            total_images = 12000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        learning_decay_rate_factor = ls["learning_decay_rate_factor"]
        num_epochs_per_decay = ls["num_epochs_per_decay"]
        NUM_GPUS = 1

        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.exponential_decay(
                learning_rate=lr * NUM_GPUS,
                decay_steps=step * num_epochs_per_decay / NUM_GPUS,
                decay_rate=learning_decay_rate_factor),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(4e-5))

    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer


def net_config(image, label, model, args):
    class_dim = args.class_dim
    model_name = args.model

    out, feature_map = model.net(input=image, class_dim=class_dim)
    cost, pred = fluid.layers.softmax_with_cross_entropy(
        out, label, return_softmax=True)
    if args.scale_loss > 1:
        avg_cost = fluid.layers.mean(x=cost) * float(args.scale_loss)
    else:
        avg_cost = fluid.layers.mean(x=cost)

    acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)

    return avg_cost, acc_top1, out, feature_map


def build_program(is_train, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model_name = args.model
    model = nets.__dict__[model_name]()
    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            if args.fp16:
                image = fluid.layers.cast(image, "float16")
            avg_cost, acc_top1, predition, feature_map = net_config(
                image, label, model, args)
            avg_cost.persistable = True
            acc_top1.persistable = True
            if is_train:
                params = model.params
                params["total_images"] = args.total_images
                params["lr"] = args.lr
                params["num_epochs"] = args.num_epochs
                params["learning_strategy"]["batch_size"] = args.batch_size
                params["learning_strategy"]["name"] = args.lr_strategy

                optimizer = optimizer_setting(params)

                if args.fp16:
                    params_grads = optimizer.backward(avg_cost)
                    master_params_grads = create_master_params_grads(
                        params_grads, main_prog, startup_prog, args.scale_loss)
                    optimizer.apply_gradients(master_params_grads)
                    master_param_to_train_param(master_params_grads,
                                                params_grads, main_prog)
                else:
                    optimizer.minimize(avg_cost)

    return py_reader, avg_cost, acc_top1, image, predition, feature_map


def train(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()

    train_py_reader, train_cost, train_acc, image, predition, feature_map = build_program(
        is_train=True,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)
    test_py_reader, test_cost, test_acc, image, predition, feature_map = build_program(
        is_train=False,
        main_prog=test_prog,
        startup_prog=startup_prog,
        args=args)
    test_prog = test_prog.clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    if args.create_module:
        assert pretrained_model, "need a pretrained module to create a hub module"
        sign1 = hub.create_signature(
            "classification", inputs=[image], outputs=[predition])
        sign2 = hub.create_signature(
            "feature_map", inputs=[image], outputs=[feature_map])
        sign3 = hub.create_signature(inputs=[image], outputs=[predition])
        hub.create_module(
            sign_arr=[sign1, sign2, sign3],
            module_dir="hub_module_" + args.model)
        exit()

    visible_device = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(['nvidia-smi',
                                              '-L']).decode().count('\n')

    train_batch_size = args.batch_size / device_num
    test_batch_size = 16
    train_reader = paddle.batch(
        reader.train(), batch_size=train_batch_size, drop_last=True)
    test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)

    train_py_reader.decorate_paddle_reader(train_reader)
    test_py_reader.decorate_paddle_reader(test_reader)
    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=bool(args.use_gpu),
        loss_name=train_cost.name)

    train_fetch_list = [train_cost.name, train_acc.name]
    test_fetch_list = [test_cost.name, test_acc.name]

    params = nets.__dict__[args.model]().params

    for pass_id in range(params["num_epochs"]):

        train_py_reader.start()

        train_info = [[], [], []]
        test_info = [[], [], []]
        train_time = []
        batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc = train_exe.run(fetch_list=train_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(np.array(loss))
                acc = np.mean(np.array(acc))
                train_info[0].append(loss)
                train_info[1].append(acc)
                train_time.append(period)
                if batch_id % 10 == 0:
                    print("Pass {0}, trainbatch {1}, loss {2}, \
                        acc {3}, time {4}".format(pass_id, batch_id, loss, acc,
                                                  "%2.2f sec" % period))
                    sys.stdout.flush()
                batch_id += 1
        except fluid.core.EOFException:
            train_py_reader.reset()

        train_loss = np.array(train_info[0]).mean()
        train_acc = np.array(train_info[1]).mean()
        train_speed = np.array(train_time).mean() / (
            train_batch_size * device_num)

        test_py_reader.start()

        test_batch_id = 0
        try:
            while True:
                t1 = time.time()
                loss, acc = exe.run(
                    program=test_prog, fetch_list=test_fetch_list)
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(loss)
                acc = np.mean(acc)
                test_info[0].append(loss)
                test_info[1].append(acc)
                if test_batch_id % 10 == 0:
                    print("Pass {0},testbatch {1},loss {2}, \
                        acc {3},time {4}".format(pass_id, test_batch_id, loss,
                                                 acc, "%2.2f sec" % period))
                    sys.stdout.flush()
                test_batch_id += 1
        except fluid.core.EOFException:
            test_py_reader.reset()

        test_loss = np.array(test_info[0]).mean()
        test_acc = np.array(test_info[1]).mean()

        print("End pass {0}, train_loss {1}, train_acc {2}, "
              "test_loss {3}, test_acc {4}".format(
                  pass_id, train_loss, train_acc, test_loss, test_acc))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path, main_program=train_prog)


def main():
    args = parser.parse_args()
    assert args.model in nets.__all__, "model is not in list %s" % nets.__all__
    print_arguments(args)
    train(args)


if __name__ == '__main__':
    main()
