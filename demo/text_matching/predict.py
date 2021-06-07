# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddlehub as hub

if __name__ == '__main__':
    data = [
        ['这个表情叫什么', '这个猫的表情叫什么'],
        ['什么是智能手环', '智能手环有什么用'],
        ['介绍几本好看的都市异能小说，要完结的！', '求一本好看点的都市异能小说，要完结的'],
        ['一只蜜蜂落在日历上（打一成语）', '一只蜜蜂停在日历上（猜一成语）'],
        ['一盒香烟不拆开能存放多久？', '一条没拆封的香烟能存放多久。'],
    ]
    label_map = {0: 'similar', 1: 'dissimilar'}

    model = hub.Module(
        name='ernie_tiny',
        version='2.0.2',
        task='text-matching',
        load_checkpoint='./checkpoint/best_model/model.pdparams',
        label_map=label_map)
    results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=True)
    for idx, texts in enumerate(data):
        print('TextA: {}\tTextB: {}\t Label: {}'.format(texts[0], texts[1], results[idx]))
