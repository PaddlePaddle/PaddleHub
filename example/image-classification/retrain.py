#-*- coding:utf8 -*-
import paddle
import paddle.fluid as fluid
import paddle_hub as hub
import paddle_hub.module as module
import sys
import reader
import argparse
import functools
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
# yapf: enable


def retrain(modelpath):

    model = module.Module(module_dir=args.hub_module_path)

    feed_list, fetch_list, program, generator = model(
        sign_name="feature_map", trainable=False)
    test_program = program.clone()
    # get the dog cat dataset
    train_reader = paddle.batch(reader.train(args.data_dir), batch_size=32)
    val_reader = paddle.batch(reader.val(args.data_dir), batch_size=32)

    with fluid.program_guard(main_program=program):
        with fluid.unique_name.guard(generator):
            img = feed_list[0]
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            feature_map = fetch_list[0]
            fc = fluid.layers.fc(input=feature_map, size=2, act="softmax")
            cost = fluid.layers.cross_entropy(input=fc, label=label)
            avg_cost = fluid.layers.mean(cost)
            acc = fluid.layers.accuracy(input=fc, label=label)

            # define the loss
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
            optimizer.minimize(avg_cost)

            # running on gpu
            place = fluid.CUDAPlace(0)
            feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
            exe = fluid.Executor(place)

            # init all param
            exe.run(fluid.default_startup_program())
            step = 0
            sample_num = 0
            epochs = 50
            # start to train
            for i in range(epochs):
                for batch in train_reader():
                    cost, accuracy = exe.run(
                        feed=feeder.feed(batch),
                        fetch_list=[avg_cost.name, acc.name])
                    step += 1
                    print(
                        "epoch %d and step %d: train cost is %.2f, train acc is %.2f%%"
                        % (i, step, cost, accuracy * 100))

            for iter, batch in enumerate(val_reader()):
                cost, accuracy = exe.run(
                    feed=feeder.feed(batch),
                    fetch_list=[avg_cost.name, acc.name])
                print("batch %d: val cost is %.2f, val acc is %.2f%%" %
                      (iter, cost, accuracy * 100))


if __name__ == "__main__":
    args = parser.parse_args()
    print_arguments(args)
    retrain(sys.argv[1])
