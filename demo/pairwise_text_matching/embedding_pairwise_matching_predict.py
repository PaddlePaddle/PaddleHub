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
"""Fine-tuning on pairwise text matching task """

import argparse
import ast
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--network", type=str, default=None, help="Pre-defined network which was connected after module.")
args = parser.parse_args()
# yapf: enable.

jieba_paddle = hub.Module(name='jieba_paddle')


def cut(text):
    res = jieba_paddle.cut(text, use_paddle=False)
    return res


if __name__ == '__main__':

    # Load Paddlehub word embedding pretrained model
    module = hub.Module(name="word2vec_skipgram")
    # module = hub.Module(name="tencent_ailab_chinese_embedding_small")
    # module = hub.Module(name="simnet_bow")

    # Pairwise task needs: query, title_left, right_title (3 slots)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len, num_slots=3)

    # Tokenizer tokenizes the text data and encodes the data as model needed.
    # If you use transformer modules (ernie, bert, roberta and so on), tokenizer should be hub.BertTokenizer.
    # Otherwise, tokenizer should be hub.CustomTokenizer.
    # If you choose CustomTokenizer, you can also change the chinese word segmentation tool, for example jieba.
    tokenizer = hub.CustomTokenizer(
        vocab_file=module.get_vocab_path(),
        tokenize_chinese_chars=True,
        cut_function=cut,  # jieba.cut as cut function
    )

    dataset = hub.dataset.DuEL(
        tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use token-level output.
    query = outputs["emb"]
    left = outputs['emb_2']
    right = outputs['emb_3']

    # Select fine-tune strategy
    strategy = hub.DefaultStrategy(optimizer_name="sgd")

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_cuda=False,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a text matching task by PaddleHub's API
    # network choice: bow, cnn, gru, lstm (PaddleHub pre-defined network)
    pairwise_matching_task = hub.PairwiseTextMatchingTask(
        dataset=dataset,
        query_feature=query,
        left_feature=left,
        right_feature=right,
        tokenizer=tokenizer,
        network=args.network,
        config=config)

    # Prediction data sample.
    text_pairs = [
        [
            "小品《战狼故事》中，吴京突破重重障碍解救爱人，深情告白太感人;爱人",  # query
            "外文名:愛人;摘要:爱人，意思是：情人。;义项描述:日本语词汇;语言:日文;中文名:爱人;标签:文化;",  # title
        ],
        [
            "儿子祝融被杀害，西天王大发雷霆，立即下令捉拿天庭三公主;儿子",  # query
            "摘要:《儿子》是曹国昌1983年创作的木雕，收藏于中国美术馆。;材质：:木雕;作者：:曹国昌;中文名:儿子;创作年代：:1983年;义项描述:曹国昌木雕;标签:文化;",  # title
        ]
    ]

    # Predict by PaddleHub's API
    results = pairwise_matching_task.predict(
        data=text_pairs,
        max_seq_len=args.max_seq_len,
        label_list=dataset.get_labels(),
        return_result=True,
        accelerate_mode=True)
    for index, text in enumerate(text_pairs):
        print("data: %s, preidction_label: %s" % (text, results[index]))
