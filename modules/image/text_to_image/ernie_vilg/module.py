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
            type="image/text_to_image",
            summary="",
            author="baidu-nlp",
            author_email="paddle-dev@baidu.com")
class ErnieVilG:

    def __init__(self):
        self.ak = 'G26BfAOLpGIRBN5XrOV2eyPA25CE01lE'
        self.sk = 'txLZOWIjEqXYMU3lSm05ViW4p9DWGOWs'
        self.token_host = 'https://wenxin.baidu.com/younger/portal/api/oauth/token'
        self.token = self._apply_token(self.ak, self.sk)

    def _apply_token(self, ak, sk):
        if ak is None or sk is None:
            ak = self.ak
            sk = self.sk
        response = requests.get(self.token_host,
                                params={
                                    'grant_type': 'client_credentials',
                                    'client_id': ak,
                                    'client_secret': sk
                                })
        if response:
            res = response.json()
            if res['code'] != 0:
                print('Request access token error.')
                raise RuntimeError("Request access token error.")
        else:
            print('Request access token error.')
            raise RuntimeError("Request access token error.")
        return res['data']

    def generate_image(self,
                       text_prompts,
                       style: Optional[str] = "油画",
                       topk: Optional[int] = 10,
                       ak: Optional[str] = None,
                       sk: Optional[str] = None,
                       output_dir: Optional[str] = 'ernievilg_output'):
        """
        Create image by text prompts using ErnieVilG model.

        :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.
        :param style: Image stype, currently supported 油画、水彩、粉笔画、卡通、儿童画、蜡笔画
        :param topk: Top k images to save.
        :param ak: ak for applying token to request wenxin api.
        :param sk: sk for applying token to request wenxin api.
        :output_dir: Output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        token = self.token
        if ak is not None and sk is not None:
            token = self._apply_token(ak, sk)

        create_url = 'https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernievilg/v1/txt2img?from=paddlehub'
        get_url = 'https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernievilg/v1/getImg?from=paddlehub'
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        taskids = []
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
                raise RuntimeError("请求参数错误")
            elif res['code'] == 4002:
                print('请求参数格式错误，请检查必传参数是否齐全，参数类型等')
                raise RuntimeError("请求参数格式错误，请检查必传参数是否齐全，参数类型等")
            elif res['code'] == 4003:
                print('请求参数中，图片风格不在可选范围内')
                raise RuntimeError("请求参数中，图片风格不在可选范围内")
            elif res['code'] == 4004:
                print('API服务内部错误，可能引起原因有请求超时、模型推理错误等')
                raise RuntimeError("API服务内部错误，可能引起原因有请求超时、模型推理错误等")
            elif res['code'] == 100 or res['code'] == 110 or res['code'] == 111:
                token = self._apply_token(ak, sk)
                res = requests.post(create_url,
                                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                    data={
                                        'access_token': token,
                                        "text": text_prompt,
                                        "style": style
                                    })
                res = res.json()
                if res['code'] != 0:
                    print("Token失效重新请求后依然发生错误，请检查输入的参数")
                    raise RuntimeError("Token失效重新请求后依然发生错误，请检查输入的参数")

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
                    raise RuntimeError("请求参数错误")
                elif res['code'] == 4002:
                    print('请求参数格式错误，请检查必传参数是否齐全，参数类型等')
                    raise RuntimeError("请求参数格式错误，请检查必传参数是否齐全，参数类型等")
                elif res['code'] == 4003:
                    print('请求参数中，图片风格不在可选范围内')
                    raise RuntimeError("请求参数中，图片风格不在可选范围内")
                elif res['code'] == 4004:
                    print('API服务内部错误，可能引起原因有请求超时、模型推理错误等')
                    raise RuntimeError("API服务内部错误，可能引起原因有请求超时、模型推理错误等")
                elif res['code'] == 100 or res['code'] == 110 or res['code'] == 111:
                    token = self._apply_token(ak, sk)
                    res = requests.post(create_url,
                                        headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                        data={
                                            'access_token': token,
                                            "text": text_prompt,
                                            "style": style
                                        })
                    res = res.json()
                    if res['code'] != 0:
                        print("Token失效重新请求后依然发生错误，请检查输入的参数")
                        raise RuntimeError("Token失效重新请求后依然发生错误，请检查输入的参数")
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
                if idx + 1 >= topk:
                    break
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
        results = self.generate_image(text_prompts=args.text_prompts,
                                      style=args.style,
                                      topk=args.topk,
                                      ak=args.ak,
                                      sk=args.sk,
                                      output_dir=args.output_dir)
        return results

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--text_prompts', type=str)
        self.arg_input_group.add_argument('--style',
                                          type=str,
                                          default='油画',
                                          choices=['油画', '水彩', '粉笔画', '卡通', '儿童画', '蜡笔画'],
                                          help="绘画风格")
        self.arg_input_group.add_argument('--topk', type=int, default=10, help="选取保存前多少张图，最多10张")
        self.arg_input_group.add_argument('--ak', type=str, default=None, help="申请文心api使用token的ak")
        self.arg_input_group.add_argument('--sk', type=str, default=None, help="申请文心api使用token的sk")
        self.arg_input_group.add_argument('--output_dir', type=str, default='ernievilg_output')
