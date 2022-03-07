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
import paddlehub as hub

if __name__ == '__main__':

    data = [
        ['这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般'],
        ['怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片'],
        ['作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。'],
    ]
    label_map = {0: 'negative', 1: 'positive'}

    model = hub.Module(
        name='ernie_tiny',
        version='2.0.1',
        task='seq-cls',
        load_checkpoint='./test_ernie_text_cls/best_model/model.pdparams',
        label_map=label_map)
    results, probs = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False, return_prob=True)
    for idx, text in enumerate(data):
        print('Data: {} \t Lable: {} \t Prob: {}'.format(text[0], results[idx], probs[idx]))
