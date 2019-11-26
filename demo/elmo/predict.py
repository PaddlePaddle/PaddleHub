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
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number in batch for training.")
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

    # Data to be prdicted
    data = [
        "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般", "交通方便；环境很好；服务态度很好 房间较小",
        "还稍微重了点，可能是硬盘大的原故，还要再轻半斤就好了。其他要进一步验证。贴的几种膜气泡较多，用不了多久就要更换了，屏幕膜稍好点，但比没有要强多了。建议配赠几张膜让用用户自己贴。",
        "前台接待太差，酒店有A B楼之分，本人check－in后，前台未告诉B楼在何处，并且B楼无明显指示；房间太小，根本不像4星级设施，下次不会再选择入住此店啦",
        "19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"
    ]

    index = 0
    run_states = elmo_task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    for batch_result in results:
        # get predict index
        batch_result = np.argmax(batch_result, axis=2)[0]
        for result in batch_result:
            print("%s\tpredict=%s" % (data[index], result))
            index += 1
