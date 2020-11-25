# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='ernie', version='2.0.0', task='sequence_classification')

    train_dataset = hub.datasets.ChnSentiCorp(
        tokenizer=model.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='train')
    dev_dataset = hub.datasets.ChnSentiCorp(
        tokenizer=model.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='dev')
    test_dataset = hub.datasets.ChnSentiCorp(
        tokenizer=model.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='test')

    optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir='test_ernie_text_cls')

    batch_size, epoch = 32, 3
    num_training_steps = len(train_dataset) / batch_size * epoch
    num_warmup_steps = 0.1 * num_training_steps
    apply_decay_param_fun = lambda x: x in [
        p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
    ]
    strategy = hub.finetune.AdamWeightDecayStrategy(
        model,
        num_training_steps,
        num_warmup_steps,
        lr=5e-5,
        weight_decay=0.01,
        apply_decay_param_fun=apply_decay_param_fun,
    )

    trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset, log_interval=10, save_interval=1)
    trainer.evaluate(test_dataset, batch_size=32)