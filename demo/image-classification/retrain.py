#-*- coding:utf8 -*-
import paddle
import paddle.fluid as fluid
import paddle_hub as hub
import paddle_hub.module as module
import sys
import os
import reader
import argparse
import functools
from visualdl import LogWriter
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('hub_module_path',  str, "hub_module_ResNet50",                  "the hub module path" )
add_arg('batch_size',       int,   32,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('num_epochs',       int,   20,                  "number of epochs.")
add_arg('class_dim',        int,   2,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('data_dir',         str,   "./dataset",  "The ImageNet dataset root dir.")
add_arg('model_save_dir',         str,   "./model_save",  "model save dir")
# yapf: enable


def retrain(modelpath):

    module = hub.Module(module_dir=args.hub_module_path)

    feed_list, fetch_list, program = module.context(
        sign_name="feature_map", trainable=True)
    # get the dog cat dataset
    train_reader = paddle.batch(reader.train(args.data_dir), batch_size=32)
    val_reader = paddle.batch(reader.val(args.data_dir), batch_size=32)

    logger = LogWriter("vdl_log", sync_cycle=5)
    with logger.mode("train") as logw:
        train_acc_scalar = logw.scalar("acc")
        train_cost_scalar = logw.scalar("cost")

    with logger.mode("val") as logw:
        val_acc_scalar = logw.scalar("acc")
        val_cost_scalar = logw.scalar("cost")

    with fluid.program_guard(main_program=program):
        img = feed_list[0]
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        feature_map = fetch_list[0]
        fc = fluid.layers.fc(input=feature_map, size=2, act="softmax")
        cost = fluid.layers.cross_entropy(input=fc, label=label)
        avg_cost = fluid.layers.mean(cost)
        acc = fluid.layers.accuracy(input=fc, label=label)
        inference_program = fluid.default_main_program().clone(for_test=True)
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        optimizer.minimize(avg_cost)

        # running on gpu
        place = fluid.CUDAPlace(0)
        feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
        exe = fluid.Executor(place)
        train_exe = fluid.ParallelExecutor(
            use_cuda=True,
            loss_name=avg_cost.name,
            main_program=fluid.default_main_program())

        # init all param
        exe.run(fluid.default_startup_program())
        step = 0
        sample_num = 0
        epochs = 50
        # start to train
        for i in range(epochs):
            train_size = 0
            train_acc = 0
            train_cost = 0
            for batch in train_reader():
                cost, accuracy = train_exe.run(
                    feed=feeder.feed(batch),
                    fetch_list=[avg_cost.name, acc.name])
                step += 1

                #####################
                train_size += 1
                train_acc += len(batch) * accuracy
                train_cost += cost
                #####################

                print(
                    "epoch %d and step %d: train cost is %.2f, train acc is %.2f%%"
                    % (i, step, cost, accuracy * 100))

            train_acc = 100 * train_acc / (train_size * 32)
            print("epoch %d: train acc is %.2f%%" % (i, train_acc))
            #####################
            train_acc_scalar.add_record(i, train_acc)
            train_cost_scalar.add_record(i, train_cost / train_size)
            #####################

            val_size = 0
            val_acc = 0
            val_cost = 0
            with fluid.program_guard(inference_program):
                for iter, batch in enumerate(val_reader()):
                    cost, accuracy = train_exe.run(
                        feed=feeder.feed(batch),
                        fetch_list=[avg_cost.name, acc.name])
                    val_size += 1
                    val_acc += len(batch) * accuracy
                    val_cost += cost
                    print("batch %d: val cost is %.2f, val acc is %.2f%%" %
                          (iter, cost, accuracy * 100))
            val_acc = 100 * val_acc / (val_size * 32)
            print("epoch %d: val acc is %.2f%%" % (i, val_acc))
            val_acc_scalar.add_record(i, val_acc)
            val_cost_scalar.add_record(i, val_cost / val_size)
            fluid.io.save_inference_model(
                dirname=os.path.join(args.model_save_dir, "iter%d" % i),
                feeded_var_names=[img.name],
                target_vars=[fc],
                executor=exe)


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    retrain(sys.argv[1])
