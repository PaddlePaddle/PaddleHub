#-*- coding:utf8 -*-
import paddle
import paddle.fluid as fluid
import paddle_hub.module as module
import reader
import sys


def retrain(modelpath):

    model = module.Module(module_dir=modelpath)

    feed_list, fetch_list, program = model(
        sign_name="feature_map", trainable=True)
    # get the dog cat dataset
    train_reader = paddle.batch(reader.train("./dataset"), batch_size=32)
    val_reader = paddle.batch(reader.val("./dataset"), batch_size=32)

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
        epochs = 50
        # start to train
        for i in range(epochs):
            for batch in train_reader():
                cost, accuracy = train_exe.run(
                    feed=feeder.feed(batch),
                    fetch_list=[avg_cost.name, acc.name])
                step += 1
                print(
                    "epoch %d and step %d: train cost is %.2f, train acc is %.2f%%"
                    % (i, step, cost, accuracy * 100))


if __name__ == "__main__":
    retrain(sys.argv[1])
