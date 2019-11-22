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
"""Finetuning on classification task """

import argparse
import paddlehub as hub
import os
from paddlehub.common.logger import logger
import shutil

# yapf: disable
parser = argparse.ArgumentParser(__doc__)

parser.add_argument("--dataset", type=str, default="chnsenticorp", help="The choice of dataset")

parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--blocks", type=int, default=0, help="dis lr blocks")
parser.add_argument("--factor", type=float, default=2.6, help="dis lr factor")

parser.add_argument("--saved_params_dir", type=str, default="", help="Directory for saving model during ")
parser.add_argument("--model_path", type=str, default="", help="load model path")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")

args = parser.parse_args()
# yapf: enable.


def is_path_valid(path):
    if path == "":
        return False
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return True


def finetune(args):
    # Download dataset and use ClassifyReader to read dataset
    if args.dataset.lower() == 'inews':
        dataset = hub.dataset.INews()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 4
        max_seq_len = 512
        num_epoch = 3
    elif args.dataset.lower().startswith("xnli"):
        dataset = hub.dataset.XNLI(language=args.dataset.lower()[-2:])
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 32
        max_seq_len = 128
        num_epoch = 2
    elif args.dataset.lower() == "lcqmc":
        dataset = hub.dataset.LCQMC()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 16
        max_seq_len = 128
        num_epoch = 3
    elif args.dataset.lower() == "tnews":
        dataset = hub.dataset.TNews()
        module = hub.Module(name="roberta_wwm_ext_chinese_L-24_H-1024_A-16")
        metrics_choices = ["acc"]
        batch_size = 16
        max_seq_len = 128
        num_epoch = 3
    else:
        raise ValueError("%s dataset is not defined" % args.dataset)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=max_seq_len)

    if args.dataset.lower() == "msraner":
        reader = hub.reader.SequenceLabelReader(
            dataset=dataset,
            vocab_path=module.get_vocab_path(),
            max_seq_len=max_seq_len)
        sequence_output = outputs["sequence_output"]
    else:
        reader = hub.reader.ClassifyReader(
            dataset=dataset,
            vocab_path=module.get_vocab_path(),
            max_seq_len=max_seq_len,
            use_task_id=False)
        pooled_output = outputs["pooled_output"]

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    scheduler = {
        "warmup": 0.1,
        "linear_decay": {
            "start_point": 0.9,
            "end_learning_rate": 0.0,
        },
        "discriminative": {
            "blocks": args.blocks,
            "factor": args.factor,
        },
    }

    # Select finetune strategy, setup config and finetune
    strategy = hub.CombinedStrategy(
        learning_rate=args.learning_rate, scheduler=scheduler)

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        log_interval=100,
        eval_interval=5000000000,
        save_ckpt_interval=100000000,
        use_data_parallel=True,
        use_pyreader=True,
        use_cuda=True,
        num_epoch=num_epoch,
        batch_size=batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a classfication finetune task by PaddleHub's API
    if args.dataset.lower() == "msraner":
        cls_task = hub.SequenceLabelTask(
            data_reader=reader,
            feature=sequence_output,
            feed_list=feed_list,
            max_seq_len=max_seq_len,
            num_classes=dataset.num_labels,
            config=config,
            add_crf=True)
    else:
        cls_task = hub.TextClassifierTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            num_classes=dataset.num_labels,
            config=config,
            metrics_choices=metrics_choices)

    # Load model from the defined model path or not
    if args.model_path != "":
        with cls_task.phase_guard(phase="train"):
            cls_task.init_if_necessary()
            cls_task.load_parameters(args.model_path)
            logger.info("PaddleHub has loaded model from %s" % args.model_path)

    # Finetune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    cls_task.finetune_and_eval()
    # run_states = cls_task.eval()
    # eval_avg_score, eval_avg_loss, eval_run_speed = cls_task._calculate_metrics(
    #     run_states)
    best_model_dir = os.path.join(config.checkpoint_dir, "best_model")

    if is_path_valid(args.saved_params_dir) and os.path.exists(best_model_dir):
        shutil.copytree(best_model_dir, args.saved_params_dir)
        shutil.rmtree(config.checkpoint_dir)

        # acc on dev will be used by auto finetune
    # print("AutoFinetuneEval" + "\t" + str())
    hub.report_final_result(cls_task.best_score)


if __name__ == "__main__":
    args = parser.parse_args()
    finetune(args)
