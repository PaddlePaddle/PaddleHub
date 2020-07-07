#coding:utf-8
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
"""Fine-tuning on classification task """

import argparse
import ast

import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size",     type=int,   default=1, help="Total examples' number in batch for training.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':
    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and get its label list and label num
    # If you just want labels information, you can omit its tokenizer parameter to avoid preprocessing the train set.
    dataset = hub.dataset.Couplet()
    num_classes = dataset.num_labels
    label_list = dataset.get_labels()

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]
    sequence_output = outputs["sequence_output"]

    # Define a classfication fine-tune task by PaddleHub's API
    gen_task = hub.GenerationTask(
        feature=pooled_output,
        token_feature=sequence_output,
        max_seq_len=args.max_seq_len,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=["bleu"])

    # Data to be predicted
    text_a = ["人增福寿年增岁", "风吹云乱天垂泪", "若有经心风过耳"]

    # Add 0x02 between characters to match the format of training data,
    # otherwise the length of prediction results will not match the input string
    # if the input string contains non-Chinese characters.
    formatted_text_a = list(map("\002".join, text_a))

    # Use the appropriate tokenizer to preprocess the data
    # For ernie_tiny, it use BertTokenizer too.
    tokenizer = hub.BertTokenizer(vocab_file=module.get_vocab_path())
    encoded_data = [
        tokenizer.encode(text=text, max_seq_len=args.max_seq_len)
        for text in formatted_text_a
    ]
    print(
        gen_task.predict(
            data=encoded_data, label_list=label_list, accelerate_mode=False))
