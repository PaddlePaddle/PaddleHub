#coding:utf-8
import argparse
import ast
import io
import numpy as np

from paddle.fluid.framework import switch_main_program
import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=5, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.05, help="Warmup proportion params for warmup strategy")
args = parser.parse_args()
# yapf: enable.


def bow_net(program, input_feature, hid_dim=128, hid_dim2=96):
    switch_main_program(program)

    bow = fluid.layers.sequence_pool(input=input_feature, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")

    return fc


def cnn_net(program, input_feature, win_size=3, hid_dim=128, hid_dim2=96):
    switch_main_program(program)

    conv_3 = fluid.nets.sequence_conv_pool(
        input=input_feature,
        num_filters=hid_dim,
        filter_size=win_size,
        act="relu",
        pool_type="max")
    fc = fluid.layers.fc(input=conv_3, size=hid_dim2)

    return fc


def gru_net(program, input_feature, hid_dim=128, hid_dim2=96):
    switch_main_program(program)

    fc0 = fluid.layers.fc(input=input_feature, size=hid_dim * 3)
    gru_h = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)
    gru_max = fluid.layers.sequence_pool(input=gru_h, pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max)
    fc = fluid.layers.fc(input=gru_max_tanh, size=hid_dim2, act='tanh')

    return fc


def bilstm_net(program, input_feature, hid_dim=128, hid_dim2=96):
    switch_main_program(program)

    fc0 = fluid.layers.fc(input=input_feature, size=hid_dim * 4)
    rfc0 = fluid.layers.fc(input=input_feature, size=hid_dim * 4)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)
    rlstm_h, c = fluid.layers.dynamic_lstm(
        input=rfc0, size=hid_dim * 4, is_reverse=True)

    # extract last step
    lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
    rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)

    lstm_last_tanh = fluid.layers.tanh(lstm_last)
    rlstm_last_tanh = fluid.layers.tanh(rlstm_last)

    # concat layer
    lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)
    # full connect layer
    fc = fluid.layers.fc(input=lstm_concat, size=hid_dim2, act='tanh')

    return fc


def lstm_net(program, input_feature, hid_dim=128, hid_dim2=96):
    switch_main_program(program)

    fc0 = fluid.layers.fc(input=input_feature, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)
    fc = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh')

    return fc


if __name__ == '__main__':
    # Step1: load Paddlehub elmo pretrained model
    module = hub.Module(name="elmo")
    inputs, outputs, program = module.context(trainable=True)

    # Step2: Download dataset and use LACClassifyReade to read dataset
    dataset = hub.dataset.ChnSentiCorp()

    reader = hub.reader.LACClassifyReader(
        dataset=dataset, vocab_path=module.get_vocab_path())
    word_dict_len = len(reader.vocab)

    word_ids = inputs["word_ids"]
    elmo_embedding = outputs["elmo_embed"]

    # Step3: switch program and build network
    # Choose the net which you would like: bow, cnn, gru, bilstm, lstm
    switch_main_program(program)

    # Embedding layer
    word_embed_dims = 128
    word_embedding = fluid.layers.embedding(
        input=word_ids,
        size=[word_dict_len, word_embed_dims],
        param_attr=fluid.ParamAttr(
            learning_rate=30,
            initializer=fluid.initializer.Uniform(low=-0.1, high=0.1)))

    # Add elmo embedding
    input_feature = fluid.layers.concat(
        input=[elmo_embedding, word_embedding], axis=1)

    # Choose the net which you would like: bow, cnn, gru, bilstm, lstm
    # We recommend you to choose the gru_net
    fc = gru_net(program, input_feature)

    # Setup feed list for data feeder
    # Must feed all the tensor of senta's module need
    feed_list = [word_ids.name]

    # Step4: Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler="linear_decay",
        warmup_proportion=args.warmup_proportion)

    # Step5: Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_cuda=args.use_gpu,
        use_data_parallel=True,
        use_pyreader=False,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Step6: Define a classfication finetune task by PaddleHub's API
    elmo_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=fc,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config)

    # Finetune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    elmo_task.finetune_and_eval()
