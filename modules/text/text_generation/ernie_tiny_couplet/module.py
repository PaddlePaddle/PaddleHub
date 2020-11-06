# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import os
import ast
import argparse

import paddlehub as hub
from paddlehub.module.module import moduleinfo, serving, runnable
from paddlehub.module.nlp_module import DataFormatError


@moduleinfo(
    name="ernie_tiny_couplet",
    version="1.0.0",
    summary="couplet generation model fine-tuned with ernie_tiny module",
    author="paddlehub",
    author_email="",
    type="nlp/text_generation",
)
class ErnieTinyCouplet(hub.NLPPredictionModule):
    def _initialize(self, use_gpu=False):
        # Load Paddlehub ERNIE Tiny pretrained model
        self.module = hub.Module(name="ernie_tiny")
        inputs, outputs, program = self.module.context(trainable=True, max_seq_len=128)

        # Download dataset and get its label list and label num
        # If you just want labels information, you can omit its tokenizer parameter to avoid preprocessing the train set.
        dataset = hub.dataset.Couplet()
        self.label_list = dataset.get_labels()

        # Setup RunConfig for PaddleHub Fine-tune API
        config = hub.RunConfig(
            use_data_parallel=False,
            use_cuda=use_gpu,
            batch_size=1,
            checkpoint_dir=os.path.join(self.directory, "assets", "ckpt"),
            strategy=hub.AdamWeightDecayStrategy())

        # Construct transfer learning network
        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_output" for token-level output.
        pooled_output = outputs["pooled_output"]
        sequence_output = outputs["sequence_output"]

        # Define a classfication fine-tune task by PaddleHub's API
        self.gen_task = hub.TextGenerationTask(
            feature=pooled_output,
            token_feature=sequence_output,
            max_seq_len=128,
            num_classes=dataset.num_labels,
            config=config,
            metrics_choices=["bleu"])

    def generate(self, texts):
        # Add 0x02 between characters to match the format of training data,
        # otherwise the length of prediction results will not match the input string
        # if the input string contains non-Chinese characters.
        formatted_text_a = list(map("\002".join, texts))

        # Use the appropriate tokenizer to preprocess the data
        # For ernie_tiny, it use BertTokenizer too.
        tokenizer = hub.BertTokenizer(vocab_file=self.module.get_vocab_path())
        encoded_data = [tokenizer.encode(text=text, max_seq_len=128) for text in formatted_text_a]
        results = self.gen_task.predict(data=encoded_data, label_list=self.label_list, accelerate_mode=False)
        results = [["".join(sample_result) for sample_result in sample_results] for sample_results in results]
        return results

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU for prediction")

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        results = self.generate(texts=input_data)

        return results

    @serving
    def serving_method(self, texts):
        """
        Run as a service.
        """
        results = self.generate(texts)
        return results


if __name__ == '__main__':
    module = ErnieTinyCouplet()
    results = module.generate(["风吹云乱天垂泪", "若有经心风过耳"])
    for result in results:
        print(result)
