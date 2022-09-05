import argparse
import random
from hashlib import md5
from typing import Optional

import requests

import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


@moduleinfo(name="baidu_translate",
            version="1.0.0",
            type="text/machine_translation",
            summary="",
            author="baidu-nlp",
            author_email="paddle-dev@baidu.com")
class BaiduTranslate:

    def __init__(self, appid=None, appkey=None):
        """
      :param appid: appid for requesting Baidu translation service.
      :param appkey: appkey for requesting Baidu translation service.
      """
        # Set your own appid/appkey.
        if appid == None:
            self.appid = '20201015000580007'
        else:
            self.appid = appid
        if appkey is None:
            self.appkey = 'IFJB6jBORFuMmVGDRud1'
        else:
            self.appkey = appkey
        self.url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

    def translate(self, query: str, from_lang: Optional[str] = "en", to_lang: Optional[int] = "zh"):
        """
        Create image by text prompts using ErnieVilG model.

        :param query: Text to be translated.
        :param from_lang: Source language.
        :param to_lang: Dst language.

        Return translated string.
        """
        # Generate salt and sign
        salt = random.randint(32768, 65536)
        sign = make_md5(self.appid + query + str(salt) + self.appkey)

        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

        # Send request
        try:
            r = requests.post(self.url, params=payload, headers=headers)
            result = r.json()
        except Exception as e:
            error_msg = str(e)
            raise RuntimeError(error_msg)
        if 'error_code' in result:
            raise RuntimeError(result['error_msg'])
        return result['trans_result'][0]['dst']

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
        if args.appid is not None and args.appkey is not None:
            self.appid = args.appid
            self.appkey = args.appkey
        result = self.translate(args.query, args.from_lang, args.to_lang)
        return result

    @serving
    def serving_method(self, query, from_lang, to_lang):
        """
        Run as a service.
        """
        return self.translate(query, from_lang, to_lang)

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--query', type=str)
        self.arg_input_group.add_argument('--from_lang', type=str, default='en', help="源语言")
        self.arg_input_group.add_argument('--to_lang', type=str, default='zh', help="目标语言")
        self.arg_input_group.add_argument('--appid', type=str, default=None, help="注册得到的个人appid")
        self.arg_input_group.add_argument('--appkey', type=str, default=None, help="注册得到的个人appkey")
