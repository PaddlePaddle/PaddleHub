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
from paddlehub.datasets.chnsenticorp import ChnSentiCorp
from paddlehub.finetune.trainer import Trainer
from paddlehub.model.modeling_ernie import ErnieforSequenceClassification
import paddle.fluid as fluid
import paddlehub as hub

if __name__ == '__main__':
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):

        ernie = hub.Module(
            directory='/mnt/zhangxuefei/program-paddle/PaddleHub/hub_module/modules/text/semantic_model/ernie_dygraph/')

        train_dataset = ChnSentiCorp(
            tokenizer=ernie.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='train')
        dev_dataset = ChnSentiCorp(
            tokenizer=ernie.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='dev')

        model = ErnieforSequenceClassification(ernie_module=ernie, num_classes=len(train_dataset.get_labels()))

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=5e-5, parameter_list=model.parameters())
        trainer = Trainer(model, optimizer, checkpoint_dir='test_ernie_text_cls')

        trainer.train(
            train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset, log_interval=10, save_interval=1)
