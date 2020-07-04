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
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--network", type=str, default=None, help="Pre-defined network which was connected after Transformer model, such as ERNIE, BERT ,RoBERTa and ELECTRA.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
parser.add_argument("--is_pair_wise", type=ast.literal_eval, default=False, help="Whether use data parallel.")


parser.add_argument("--module_name", type=str, default='ernie', help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':

    # Load Paddlehub ERNIE Tiny pretrained model
    module = hub.Module(name=args.module_name)
    if args.is_pair_wise:
        num_data = 3
    else:
        num_data = 2
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len, num_data=num_data)

    if 'ernie' in args.module_name:
        tokenizer = hub.BertTokenizer(
            vocab_file=module.get_vocab_path(), tokenize_chinese_chars=True)
        query = outputs["sequence_output"]
        left = outputs['sequence_output_2']
        right = outputs['sequence_output_3']
    else:
        tokenizer = hub.CustomTokenizer(
            vocab_file=module.get_vocab_path(), tokenize_chinese_chars=True)
        query = outputs["emb_1"]
        left = outputs['emb_2']
        right = outputs['emb_3']

    if args.is_pair_wise:
        dataset = hub.dataset.DuEL(
            tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    else:
        dataset = hub.dataset.LCQMC(
            tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use "emb" for token-level output.

    # Select fine-tune strategy, setup config and fine-tune
    strategy = hub.DefaultStrategy(
        optimizer_name="sgd",
        learning_rate=args.learning_rate)  # lazy_mode=True)

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        eval_interval=10,
        use_cuda=args.use_gpu,
        use_data_parallel=args.use_data_parallel,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a classfication fine-tune task by PaddleHub's API
    matching_task = hub.TextMatchingTask(
        query_feature=query,
        left_feature=left,
        tokenizer=tokenizer,
        network=args.network,
        is_pair_wise=args.is_pair_wise,
        right_feature=right if args.is_pair_wise else None,
        config=config)

    # Data to be predicted
    if args.is_pair_wise:
        # pair_wise prediction data
        text_pairs = [
            [
                "思追原来是个超级妹控，不愿妹妹嫁人，然而妹妹却喜欢一博老师;老师",  # text_1
                "摘要:林妙可演唱的《老师》，作为中央电视台教师节晚会的开场曲。;谱曲:彭野新儿歌;歌曲时长:3分53秒;",  # text_2
                "摘要:林妙可演唱的《老师》，作为中央电视台教师节晚会的开场曲。;谱曲:彭野新儿歌;歌曲时长:3分53秒;"  # text_3 same as text_2
            ],
            [
                "儿子祝融被杀害，西天王大发雷霆，立即下令捉拿天庭三公主;儿子",  # text_1
                "摘要:《儿子》是曹国昌1983年创作的木雕，收藏于中国美术馆。;材质：:木雕;作者：:曹国昌;中文名:儿子;创作年代：:1983年;义项描述:曹国昌木雕;标签:文化;",  # text_2
                "摘要:《儿子》是曹国昌1983年创作的木雕，收藏于中国美术馆。;材质：:木雕;作者：:曹国昌;中文名:儿子;创作年代：:1983年;义项描述:曹国昌木雕;标签:文化;",  # text_3 same as text_2
            ]
        ]
    else:
        # point_wise prediction data
        text_pairs = [
            # point_wise prediction data
            [
                "请问不是您的账户吗？",  # text_1
                "您好，请问您使用的邮箱类型是？"  # text_2
            ],
            [
                "推荐个手机游戏",  # text_1
                "手机游戏推荐"  # text_2
            ]
        ]

    # Fine-tune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    results = matching_task.predict(
        data=text_pairs,
        max_seq_len=args.max_seq_len,
        label_list=dataset.get_labels(),
        return_result=True,
        accelerate_mode=False)
    print(results)
    for index, text in enumerate(text_pairs):
        print("data: %s, preidction_label: %s" % (text, results[index]))
