import json
import argparse
from typing import List

import requests
from paddlehub.module.module import moduleinfo, runnable


def get_access_token(api_key: str, secret_key: str) -> str:
    '''
    Get Access Token

    Params:
        api_key(str): API Key
        secret_key(str): Secret Key

    Return:
        access_token(str): Access Token
    '''
    url = 'https://wenxin.baidu.com/younger/portal/api/oauth/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    datas = {
        'grant_type': 'client_credentials',
        'client_id': api_key if api_key != '' else 'G26BfAOLpGIRBN5XrOV2eyPA25CE01lE',
        'client_secret': secret_key if secret_key != '' else 'txLZOWIjEqXYMU3lSm05ViW4p9DWGOWs'
    }

    results = json.loads(requests.post(url, datas, headers=headers).text)

    assert results['msg'] == 'success', f"Error message: '{results['msg']}'. Please check the api_key and secret_key."

    return results['data']


@moduleinfo(
    name='ERNIE3_Zeus',
    type='nlp/text_generation',
    author='paddlepaddle',
    author_email='',
    summary='ERNIE3_Zeus',
    version='1.0.0'
)
class ERNIE3Zeus:
    @staticmethod
    def custom_generation(text: str, seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = '', topk: int = 10,
                          temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                          is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 自定义接口

        Params:
            text(srt): 输入内容, 长度不超过 1000。
            seq_len(int): 输出内容最大长度, 长度不超过 1000。
            task_prompt(str): 任务类型的模板。
            dataset_prompt(str): 数据集类型的模板。
            topk(int): topk采样, 取值 > 1, 默认为 10。每步的生成的结果从 topk 的概率值分布中采样。其中 topk = 1 表示贪婪采样, 每次生成结果固定。
            temperature(float): 温度系数, 取值 > 0.0, 默认为 1.0。更大的温度系数表示模型生成的多样性更强。
            penalty_score(float): 重复惩罚。取值 >= 1.0, 默认为 1.0。通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。
            penalty_text(str): 惩罚文本, 默认为空。模型无法生成该字符串中的 token。通过设置该值, 可以减少某些冗余与异常字符的生成。
            choice_text(str): 候选文本, 默认为空。模型只能生成该字符串中的 token 的组合。通过设置该值, 可以对某些抽取式任务进行定向调优。
            stop_token(str): 提前结束符, 默认为空。预测结果解析时使用的结束字符, 碰到对应字符则直接截断并返回。可以通过设置该值, 过滤掉 few-shot 等场景下模型重复的 cases。
            is_unidirectional(bool): 单双向控制开关, 取值 0 或者 1, 默认为 0。0 表示模型为双向生成, 1 表示模型为单向生成。建议续写与 few-shot 等通用场景建议采用单向生成方式, 而完型填空等任务相关场景建议采用双向生成方式。
            min_dec_len(int): 最小生成长度, 取值 >= 1, 默认为 1。开启后会屏蔽掉 END 结束符号, 让模型生成至指定的最小长度。
            min_dec_penalty_text(str): 默认为空, 与最小生成长度搭配使用, 可以在 min_dec_len 步前不让模型生成该字符串中的 tokens。
            api_key(str): API Key。
            secret_key(str): Secret Key。

        Return: 
            text(str): 生成的文本
        '''
        access_token = get_access_token(api_key, secret_key)

        url = 'https://wenxin.baidu.com/younger/portal/api/rest/1.0/ernie/3.0/zeus'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        datas = {
            'access_token': access_token,
            'text': text,
            'seq_len': seq_len,
            'task_prompt': task_prompt,
            'dataset_prompt': dataset_prompt,
            'topk': topk,
            'temperature': temperature,
            'penalty_score': penalty_score,
            'penalty_text': penalty_text,
            'choice_text': choice_text,
            'stop_token': stop_token,
            'is_unidirectional': int(is_unidirectional),
            'min_dec_len': min_dec_len,
            'min_dec_penalty_text': min_dec_penalty_text
        }

        info = datas.copy()
        info.pop('access_token')
        print(json.dumps(info, ensure_ascii=False, indent=4))

        results = json.loads(requests.post(url, datas, headers=headers).text)

        assert results['code'] == 0, f"Error message: '{results['msg']}'."

        return results['data']['result']

    def article_creation(self, text: List[str], seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = 'zuowen', topk: int = 10,
                         temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                         is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 作文创作
        '''
        assert len(text) == 2, 'text num should be equal to 2.'
        text = f'作文题目：{text[0]} 题目内容：{text[1]} 作文内容：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def copywriting_creation(self, text: str, seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = '', topk: int = 10,
                             temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                             is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 文案创作
        '''
        text = f'产品：{text} 文案：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def text_summarization(self, text: str, seq_len: int = 256, task_prompt: str = 'Summarization', dataset_prompt: str = '', topk: int = 10,
                           temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                           is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 摘要生成
        '''
        text = f'请给下面这段话写一句摘要："{text}"'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def question_generation(self, text: str, seq_len: int = 256, task_prompt: str = 'QuestionGeneration', dataset_prompt: str = '', topk: int = 10,
                            temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                            is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 问题生成
        '''
        text = f'文本：{text} 提问：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def poetry_creation(self, text: str, seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = 'poetry', topk: int = 10,
                        temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                        is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 古诗创作
        '''
        assert len(text) == 2, 'text num should be equal to 2.'
        text = f'古诗题目：{text[0]} 内容：{text[1]}'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def couplet_continuation(self, text: str, seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = 'couplet', topk: int = 10,
                             temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                             is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 对联续写
        '''
        text = f'上联:{text} 下联:'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def answer_generation(self, text: str, seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = '', topk: int = 10,
                          temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                          is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 自由问答
        '''
        text = f'问题：{text} 回答：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def article_continuation(self, text: str, seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = '', topk: int = 10,
                             temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                             is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 小说续写
        '''
        text = f'上文：{text} 下文：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def sentiment_classification(self, text: str, seq_len: int = 256, task_prompt: str = 'SentimentClassification', dataset_prompt: str = '', topk: int = 10,
                                 temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                                 is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 情感分析
        '''
        text = f'下面这个评价是正面还是负面的？"{text}"'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def information_extraction(self, text: List[str], seq_len: int = 256, task_prompt: str = 'QA_MRC', dataset_prompt: str = '', topk: int = 10,
                               temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                               is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 信息抽取
        '''
        assert len(text) == 2, 'text num should be equal to 2.'
        text = f'"{text[0]}"，问："{text[1]}"，答：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def synonymous_rewriting(self, text: str, seq_len: int = 256, task_prompt: str = 'Paraphrasing', dataset_prompt: str = '', topk: int = 10,
                             temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                             is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 同义改写
        '''
        text = f'"{text}"的另一种说法是：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def semantic_matching(self, text: List[str], seq_len: int = 256, task_prompt: str = 'SemanticMatching', dataset_prompt: str = '', topk: int = 10,
                          temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                          is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 文本匹配
        '''
        assert len(text) == 2, 'text num should be equal to 2.'
        text = f'如果说"{text[0]}"可不可以认为"{text[1]}"？ '
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def text_correction(self, text: str, seq_len: int = 256, task_prompt: str = 'Correction', dataset_prompt: str = '', topk: int = 10,
                        temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                        is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 文本纠错
        '''
        text = f'改正下面文本中的错误：“{text}”'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def text_cloze(self, text: str, seq_len: int = 256, task_prompt: str = '', dataset_prompt: str = '', topk: int = 10,
                   temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                   is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 完形填空
        '''
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    def text2SQL(self, text: str, seq_len: int = 256, task_prompt: str = 'Text2SQL', dataset_prompt: str = '', topk: int = 10,
                 temperature: float = 1.0, penalty_score: float = 1.0, penalty_text: str = '',  choice_text: str = '', stop_token: str = '',
                 is_unidirectional: bool = False, min_dec_len: int = 1, min_dec_penalty_text: str = '', api_key: str = '', secret_key: str = '') -> str:
        '''
        ERNIE 3.0 Zeus 文本转 SQL 语句
        '''
        text = f'问题：{text}。 把这个问题转化成SQL语句：'
        return self.custom_generation(
            text, seq_len, task_prompt, dataset_prompt, topk,
            temperature, penalty_score, penalty_text,  choice_text, stop_token,
            is_unidirectional, min_dec_len, min_dec_penalty_text, api_key, secret_key
        )

    @runnable
    def cmd(self, argvs):
        parser = argparse.ArgumentParser(
            description="Run the {}".format(self.name),
            prog="hub run {}".format(self.name),
            usage='%(prog)s',
            add_help=True)

        parser.add_argument('--task', type=str, default='custom_generation')
        parser.add_argument('--text', type=str, required=True, nargs='+')
        parser.add_argument('--seq_len', type=int, default=256)
        parser.add_argument('--task_prompt', type=str, default='')
        parser.add_argument('--dataset_prompt', type=str, default='')
        parser.add_argument('--topk', type=int, default=10)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--penalty_score', type=float, default=1.0)
        parser.add_argument('--penalty_text', type=str, default='')
        parser.add_argument('--choice_text', type=str, default='')
        parser.add_argument('--stop_token', type=str, default='')
        parser.add_argument('--is_unidirectional', type=bool, default=False)
        parser.add_argument('--min_dec_len', type=int, default=1)
        parser.add_argument('--min_dec_penalty_text', type=str, default='')
        parser.add_argument('--api_key', type=str, default='')
        parser.add_argument('--secret_key', type=str, default='')

        args = parser.parse_args(argvs)

        func = getattr(self, args.task)

        kwargs = vars(args)
        kwargs.pop('task')

        if len(kwargs['text']) == 1:
            kwargs['text'] = kwargs['text'][0]

        return func(**kwargs)
