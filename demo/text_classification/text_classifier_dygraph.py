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
parser.add_argument("--batch_size",         type=int,               default=16,                                 help="Total examples' number in batch for training.")
parser.add_argument("--log_interval",       type=int,               default=10,                                 help="log interval.")
parser.add_argument("--save_interval",      type=int,               default=10,                                 help="save interval.")
parser.add_argument("--checkpoint_dir",     type=str,               default="paddlehub_finetune_ckpt_dygraph",  help="Path to save log data.")
parser.add_argument("--max_seq_len",        type=int,               default=512,                                help="Number of words of the longest seqence.")
# yapf: enable.


class TransformerClassifier(fluid.dygraph.Layer):
    def __init__(self, num_classes, transformer):
        super(TransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.transformer = transformer
        self.fc = Linear(input_dim=768, output_dim=num_classes)

    def forward(self, input_ids, position_ids, segment_ids, input_mask):
        result = self.transformer(input_ids, position_ids, segment_ids,
                                  input_mask)
        cls_feats = fluid.layers.dropout(
            result['pooled_output'],
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        cls_feats = fluid.layers.reshape(cls_feats, shape=[-1, 768])
        pred = self.fc(cls_feats)
        return fluid.layers.softmax(pred)


def finetune(args):
    module = hub.Module(name="ernie", max_seq_len=args.max_seq_len)
    # Use the appropriate tokenizer to preprocess the data set
    # For ernie_tiny, it will do word segmentation to get subword. More details: https://www.jiqizhixin.com/articles/2019-11-06-9
    if module.name == "ernie_tiny":
        tokenizer = hub.ErnieTinyTokenizer(
            vocab_file=module.get_vocab_path(),
            spm_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path(),
        )
    else:
        tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path())
    dataset = hub.dataset.ChnSentiCorp(
        tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    with fluid.dygraph.guard():
        tc = TransformerClassifier(
            num_classes=dataset.num_labels, transformer=module)
        adam = AdamOptimizer(learning_rate=1e-5, parameter_list=tc.parameters())
        state_dict_path = os.path.join(args.checkpoint_dir,
                                       'dygraph_state_dict')
        if os.path.exists(state_dict_path + '.pdparams'):
            state_dict, _ = fluid.load_dygraph(state_dict_path)
            tc.load_dict(state_dict)

        loss_sum = acc_sum = cnt = 0
        for epoch in range(args.num_epoch):
            for batch_id, data in enumerate(
                    dataset.batch_records_generator(
                        phase="train",
                        batch_size=args.batch_size,
                        shuffle=True,
                        pad_to_batch_max_seq_len=False)):
                batch_size = len(data["input_ids"])
                input_ids = np.array(data["input_ids"]).astype(
                    np.int64).reshape([batch_size, -1, 1])
                position_ids = np.array(data["position_ids"]).astype(
                    np.int64).reshape([batch_size, -1, 1])
                segment_ids = np.array(data["segment_ids"]).astype(
                    np.int64).reshape([batch_size, -1, 1])
                input_mask = np.array(data["input_mask"]).astype(
                    np.float32).reshape([batch_size, -1, 1])
                labels = np.array(data["label"]).astype(np.int64).reshape(
                    [batch_size, 1])
                pred = tc(input_ids, position_ids, segment_ids, input_mask)

                acc = fluid.layers.accuracy(pred, to_variable(labels))
                loss = fluid.layers.cross_entropy(pred, to_variable(labels))
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)

                loss_sum += avg_loss.numpy() * labels.shape[0]
                acc_sum += acc.numpy() * labels.shape[0]
                cnt += labels.shape[0]
                if batch_id % args.log_interval == 0:
                    print('epoch {}: loss {}, acc {}'.format(
                        epoch, loss_sum / cnt, acc_sum / cnt))
                    loss_sum = acc_sum = cnt = 0

                if batch_id % args.save_interval == 0:
                    state_dict = tc.state_dict()
                    fluid.save_dygraph(state_dict, state_dict_path)


if __name__ == "__main__":
    args = parser.parse_args()
    finetune(args)
