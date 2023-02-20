import argparse
import os
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


@moduleinfo(
    name="baidu_language_recognition",
    version="1.1.0",
    type="text/machine_translation",
    summary="",
    author="baidu-nlp",
    author_email="paddle-dev@baidu.com")
class BaiduLanguageRecognition:
    def __init__(self, appid=None, appkey=None):
        """
      :param appid: appid for requesting Baidu translation service.
      :param appkey: appkey for requesting Baidu translation service.
      """
        # Set your own appid/appkey.
        if appid == None:
            self.appid = os.getenv('BT_APPID')
        else:
            self.appid = appid
        if appkey is None:
            self.appkey = os.getenv('BT_APPKEY')
        else:
            self.appkey = appkey
        if self.appid is None and self.appkey is None:
            raise RuntimeError("Please set appid and appkey.")
        self.url = 'https://fanyi-api.baidu.com/api/trans/vip/language'

    def recognize(self, query: str):
        """
        Create image by text prompts using ErnieVilG model.

        :param query: Text to be translated.

        Return language type code.
        """
        # Generate salt and sign
        salt = random.randint(32768, 65536)
        sign = make_md5(self.appid + query + str(salt) + self.appkey)

        # Build request
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': self.appid, 'q': query, 'salt': salt, 'sign': sign}

        # Send request
        try:
            r = requests.post(self.url, params=payload, headers=headers)
            result = r.json()
        except Exception as e:
            error_msg = str(e)
            raise RuntimeError(error_msg)
        if result['error_code'] != 0:
            raise RuntimeError(result['error_msg'])
        return result['data']['src']

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
            usage='%(prog)s',
            add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        if args.appid is not None and args.appkey is not None:
            self.appid = args.appid
            self.appkey = args.appkey
        result = self.recognize(args.query)
        return result

    @serving
    def serving_method(self, query):
        """
        Run as a service.
        """
        return self.recognize(query)

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--query', type=str)
        self.arg_input_group.add_argument('--appid', type=str, default=None, help="注册得到的个人appid")
        self.arg_input_group.add_argument('--appkey', type=str, default=None, help="注册得到的个人appkey")
