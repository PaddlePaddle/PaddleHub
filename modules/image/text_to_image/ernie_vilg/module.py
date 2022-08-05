# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import ast
import os
import re
import sys
import time
from functools import partial
from io import BytesIO
from typing import List
from typing import Optional

import requests
from PIL import Image
from tqdm.auto import tqdm

import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="ernie_vilg",
            version="1.0.0",
            type="MultiModal/image_generation",
            summary="",
            author="paddlepaddle",
            author_email="paddle-dev@baidu.com")
class ErnieVilG:

    def generate_image(self,
                       text_prompts: Optional[List[str]] = ["宁静的乡村"],
                       style: Optional[str] = "油画",
                       output_dir: Optional[str] = 'ernievilg_output'):
        """
        Create image by text prompts using ErnieVilG model.

        :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.
        :param style: Image stype, currently supported 油画、水彩画、中国画
        :output_dir: Output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        ak = 'G26BfAOLpGIRBN5XrOV2eyPA25CE01lE'
        sk = 'txLZOWIjEqXYMU3lSm05ViW4p9DWGOWs'
        token_host = 'https://wenxin.baidu.com/younger/portal/api/oauth/token'
        response = requests.get(token_host,
                                params={
                                    'grant_type': 'client_credentials',
                                    'client_id': ak,
                                    'client_secret': sk
                                })
        if response:
            res = response.json()
            if res['code'] != 0:
                print('Request access token error.')
                exit(-1)
        else:
            print('Request access token error.')
            exit(-1)

        token = res['data']
        create_url = 'https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernievilg/v1/txt2img'
        get_url = 'https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernievilg/v1/getImg'
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        taskids = []
        error = False
        for text_prompt in text_prompts:
            res = requests.post(create_url,
                                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                data={
                                    'access_token': token,
                                    "text": text_prompt,
                                    "style": style
                                })
            res = res.json()
            if res['code'] == 4001:
                print('请求参数错误')
                error = True
            elif res['code'] == 4002:
                print('请求参数格式错误，请检查必传参数是否齐全，参数类型等')
                error = True
            elif res['code'] == 4003:
                print('请求参数中，图片风格不在可选范围内')
                error = True
            elif res['code'] == 4004:
                print('API服务内部错误，可能引起原因有请求超时、模型推理错误等')
                error = True
            if error == True:
                exit(-1)
            taskids.append(res['data']["taskId"])

        start_time = time.time()
        process_bar = tqdm(total=100, unit='%')
        results = {}
        first_iter = True
        while True:
            if not taskids:
                break
            total_time = 0
            has_done = []
            for taskid in taskids:
                res = requests.post(get_url,
                                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                    data={
                                        'access_token': token,
                                        'taskId': {taskid}
                                    })
                res = res.json()
                if res['code'] == 4001:
                    print('请求参数错误')
                    error = True
                elif res['code'] == 4002:
                    print('请求参数格式错误，请检查必传参数是否齐全，参数类型等')
                    error = True
                elif res['code'] == 4003:
                    print('请求参数中，图片风格不在可选范围内')
                    error = True
                elif res['code'] == 4004:
                    print('API服务内部错误，可能引起原因有请求超时、模型推理错误等')
                    error = True
                if error == True:
                    exit(-1)
                if res['data']['status'] == 1:
                    has_done.append(res['data']['taskId'])
                results[res['data']['text']] = {
                    'imgUrls': res['data']['imgUrls'],
                    'waiting': res['data']['waiting'],
                    'taskId': res['data']['taskId']
                }
                total_time = int(re.match('[0-9]+', str(res['data']['waiting'])).group(0)) * 60
            end_time = time.time()
            progress_rate = int(((end_time - start_time) / total_time * 100)) if total_time != 0 else 100
            if progress_rate > process_bar.n:
                increase_rate = progress_rate - process_bar.n
                if progress_rate >= 100:
                    increase_rate = 100 - process_bar.n
            else:
                increase_rate = 0
            process_bar.update(increase_rate)
            time.sleep(5)
            for taskid in has_done:
                taskids.remove(taskid)
        print('Saving Images...')
        result_images = []
        for text, data in results.items():
            for idx, imgdata in enumerate(data['imgUrls']):
                image = Image.open(BytesIO(requests.get(imgdata['image']).content))
                image.save(os.path.join(output_dir, '{}_{}.png'.format(text, idx)))
                result_images.append(image)
        print('Done')
        return result_images

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.generate_image(text_prompts=args.text_prompts, style=args.style, output_dir=args.output_dir)
        return results

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--text_prompts', type=str, default='宁静的小镇')
        self.arg_input_group.add_argument('--style', type=str, default='油画', choices=['油画', '水彩画', '中国画'], help="绘画风格")
        self.arg_input_group.add_argument('--output_dir', type=str, default='ernievilg_output')
