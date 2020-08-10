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
"""Fine-tuning on ponitwise text matching task """

import argparse
import ast
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':

    # Load Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")

    # Pointwise task needs: query, title_left (2 slots)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len, num_slots=2)

    # Tokenizer tokenizes the text data and encodes the data as model needed.
    # If you use transformer modules (ernie, bert, roberta and so on), tokenizer should be hub.BertTokenizer.
    # else tokenizer should be hub.CustomTokenizer.
    tokenizer = hub.BertTokenizer(
        vocab_file=module.get_vocab_path(), tokenize_chinese_chars=True)

    # Load dataset
    dataset = hub.dataset.LCQMC(
        tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use token-level output.
    query = outputs["sequence_output"]
    left = outputs['sequence_output_2']

    # Select fine-tune strategy
    strategy = hub.AdamWeightDecayStrategy()

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a pointwise text matching task by PaddleHub's API
    pointwise_matching_task = hub.PointwiseTextMatchingTask(
        dataset=dataset,
        query_feature=query,
        title_feature=left,
        tokenizer=tokenizer,
        config=config)

    # Prediction data sample.
    text_pairs = [
        [
            "淘宝上怎么用信用卡分期付款",  # query
            "淘宝上怎么分期付款，没有信用卡",  # title
        ],
        [
            "山楂干怎么吃好吃？",  # query
            "山楂怎么做好吃",  # title
        ]
    ]

    # Predict by PaddleHub's API
    results = pointwise_matching_task.predict(
        data=text_pairs,
        max_seq_len=args.max_seq_len,
        label_list=dataset.get_labels(),
        return_result=True,
        accelerate_mode=True)
    for index, text in enumerate(text_pairs):
        print("data: %s, prediction_label: %s" % (text, results[index]))
