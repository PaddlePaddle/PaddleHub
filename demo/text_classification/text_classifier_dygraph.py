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


def finetune(args):
    with fluid.dygraph.guard():
        ernie = hub.Module(name="ernie")
        dataset = hub.dataset.ChnSentiCorp()

        reader = hub.reader.ClassifyReader(
            dataset=dataset,
            vocab_path=ernie.get_vocab_path(),
            max_seq_len=args.max_seq_len,
            sp_model_path=ernie.get_spm_path(),
            word_dict_path=ernie.get_word_dict_path())
        train_reader = reader.data_generator(
            batch_size=args.batch_size, phase='train')

        for data_id, data in enumerate(train_reader()):
            input_ids = np.array(data[0][0]).astype(np.int64)
            position_ids = np.array(data[0][1]).astype(np.int64)
            segment_ids = np.array(data[0][2]).astype(np.int64)
            input_mask = np.array(data[0][3]).astype(np.float32)
            labels = np.array(data[0][4]).astype(np.int64)
            pooled_output, sequence_output = ernie(position_ids, input_mask,
                                                   input_ids, segment_ids)


if __name__ == "__main__":
    args = parser.parse_args()
    finetune(args)
