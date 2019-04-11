#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning on classification task """

import argparse

import paddle.fluid as fluid
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--data_dir", type=str, default=None, help="Path to training data.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Step1: load Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Step2: Download dataset and use ClassifyReader to read dataset
    reader = hub.reader.ClassifyReader(
        dataset=hub.dataset.LCQMC(),
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len)
    num_labels = len(reader.get_labels())

    # Step3: construct transfer learning network
    with fluid.program_guard(program):
        label = fluid.layers.data(name="label", shape=[1], dtype='int64')

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_output" for token-level output.
        pooled_output = outputs["pooled_output"]

        # Setup feed list for data feeder
        # Must feed all the tensor of ERNIE's module need
        feed_list = [
            inputs["input_ids"].name, inputs["position_ids"].name,
            inputs["segment_ids"].name, inputs["input_mask"].name, label.name
        ]
        # Define a classfication finetune task by PaddleHub's API
        cls_task = hub.create_text_classification_task(
            pooled_output, label, num_classes=num_labels)

        # Step4: Select finetune strategy, setup config and finetune
        strategy = hub.BERTFinetuneStrategy(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            warmup_strategy="linear_warmup_decay",
        )

        # Setup runing config for PaddleHub Finetune API
        config = hub.RunConfig(
            use_cuda=True,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            strategy=strategy)

        # Finetune and evaluate by PaddleHub's API
        # will finish training, evaluation, testing, save model automatically
        hub.finetune_and_eval(
            task=cls_task,
            data_reader=reader,
            feed_list=feed_list,
            config=config)
