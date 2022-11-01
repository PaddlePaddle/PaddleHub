import json
import argparse

import requests
from paddlehub.module.module import moduleinfo, runnable


def get_access_token(ak: str = '', sk: str = '') -> str:
    '''
    Get Access Token

    Params:
        ak(str): API Key
        sk(str): Secret Key

    Return:
        access_token(str): Access Token
    '''
    url = 'https://wenxin.baidu.com/younger/portal/api/oauth/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    datas = {
        'grant_type': 'client_credentials',
        'client_id': ak if ak != '' else 'G26BfAOLpGIRBN5XrOV2eyPA25CE01lE',
        'client_secret': sk if sk != '' else 'txLZOWIjEqXYMU3lSm05ViW4p9DWGOWs'
    }

    responses = requests.post(url, datas, headers=headers)

    assert responses.status_code == 200, f"Network Error {responses.status_code}."

    results = json.loads(responses.text)

    assert results['msg'] == 'success', f"Error message: '{results['msg']}'. Please check the ak and sk."

    return results['data']


@moduleinfo(
    name='ernie_zeus',
    type='nlp/text_generation',
    author='paddlepaddle',
    author_email='',
    summary='ernie_zeus',
    version='1.0.0'
)
class ERNIEZeus:
    def __init__(self, ak: str = '', sk: str = '') -> None:
        self.access_token = get_access_token(ak, sk)

    def custom_generation(self,
                          text: str,
                          min_dec_len: int = 1,
                          seq_len: int = 128,
                          topp: float = 1.0,
                          penalty_score: float = 1.0,
                          stop_token: str = '',
                          task_prompt: str = '',
                          penalty_text: str = '',
                          choice_text: str = '',
                          is_unidirectional: bool = False,
                          min_dec_penalty_text: str = '',
                          logits_bias: int = -10000,
                          mask_type: str = 'word') -> str:
        '''
        ERNIE 3.0 Zeus 自定义接口

        Params:
            text(srt): 模型的输入文本, 为 prompt 形式的输入。文本长度 [1, 1000]。注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512。
            min_dec_len(int): 输出结果的最小长度, 避免因模型生成 END 或者遇到用户指定的 stop_token 而生成长度过短的情况,与 seq_len 结合使用来设置生成文本的长度范围 [1, seq_len]。
            seq_len(int): 输出结果的最大长度, 因模型生成 END 或者遇到用户指定的 stop_token, 实际返回结果可能会小于这个长度, 与 min_dec_len 结合使用来控制生成文本的长度范围 [1, 1000]。(注: ERNIE 3.0-1.5B 模型取值范围 ≤ 512)
            topp(float): 影响输出文本的多样性, 取值越大, 生成文本的多样性越强。取值范围 [0.0, 1.0]。
            penalty_score(float): 通过对已生成的 token 增加惩罚, 减少重复生成的现象。值越大表示惩罚越大。取值范围 [1.0, 2.0]。
            stop_token(str): 预测结果解析时使用的结束字符串, 碰到对应字符串则直接截断并返回。可以通过设置该值, 过滤掉 few-shot 等场景下模型重复的 cases。
            task_prompt(str): 指定预置的任务模板, 效果更好。
                              PARAGRAPH: 引导模型生成一段文章; SENT: 引导模型生成一句话; ENTITY: 引导模型生成词组; 
                              Summarization: 摘要; MT: 翻译; Text2Annotation: 抽取; Correction: 纠错; 
                              QA_MRC: 阅读理解; Dialogue: 对话; QA_Closed_book: 闭卷问答; QA_Multi_Choice: 多选问答; 
                              QuestionGeneration: 问题生成; Paraphrasing: 复述; NLI: 文本蕴含识别; SemanticMatching: 匹配; 
                              Text2SQL: 文本描述转SQL; TextClassification: 文本分类; SentimentClassification: 情感分析; 
                              zuowen: 写作文; adtext: 写文案; couplet: 对对联; novel: 写小说; cloze: 文本补全; Misc: 其它任务。
            penalty_text(str): 模型会惩罚该字符串中的 token。通过设置该值, 可以减少某些冗余与异常字符的生成。
            choice_text(str): 模型只能生成该字符串中的 token 的组合。通过设置该值, 可以对某些抽取式任务进行定向调优。
            is_unidirectional(bool): False 表示模型为双向生成, True 表示模型为单向生成。建议续写与 few-shot 等通用场景建议采用单向生成方式, 而完型填空等任务相关场景建议采用双向生成方式。
            min_dec_penalty_text(str): 与最小生成长度搭配使用, 可以在 min_dec_len 步前不让模型生成该字符串中的 tokens。
            logits_bias(int): 配合 penalty_text 使用, 对给定的 penalty_text 中的 token 增加一个 logits_bias, 可以通过设置该值屏蔽某些 token 生成的概率。
            mask_type(str): 设置该值可以控制模型生成粒度。可选参数为 word, sentence, paragraph。

        Return: 
            text(str): 生成的文本
        '''
        url = 'https://wenxin.baidu.com/moduleApi/portal/api/rest/1.0/ernie/3.0.28/zeus?from=paddlehub'
        access_token = self.access_token
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        datas = {
            'access_token': access_token,
            'text': text,
            'min_dec_len': min_dec_len,
            'seq_len': seq_len,
            'topp': topp,
            'penalty_score': penalty_score,
            'stop_token': stop_token,
            'task_prompt': task_prompt,
            'penalty_text': penalty_text,
            'choice_text': choice_text,
            'is_unidirectional': int(is_unidirectional),
            'min_dec_penalty_text': min_dec_penalty_text,
            'logits_bias': logits_bias,
            'mask_type': mask_type,
        }

        responses = requests.post(url, datas, headers=headers)

        assert responses.status_code == 200, f"Network Error {responses.status_code}."

        results = json.loads(responses.text)

        assert results['code'] == 0, f"Error message: '{results['msg']}'."

        return results['data']['result']

    def text_generation(self,
                        text: str,
                        min_dec_len: int = 4,
                        seq_len: int = 512,
                        topp: float = 0.9,
                        penalty_score: float = 1.2) -> str:
        '''
        文本生成
        '''
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='PARAGRAPH',
            penalty_text='[{[gEND]',
            choice_text='',
            is_unidirectional=True,
            min_dec_penalty_text='。？：！[<S>]',
            logits_bias=-10,
            mask_type='paragraph'
        )

    def text_summarization(self,
                           text: str,
                           min_dec_len: int = 4,
                           seq_len: int = 512,
                           topp: float = 0.0,
                           penalty_score: float = 1.0) -> str:
        '''
        摘要生成
        '''
        text = "文章：{} 摘要：".format(text)
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='Summarization',
            penalty_text='',
            choice_text='',
            is_unidirectional=False,
            min_dec_penalty_text='',
            logits_bias=-10000,
            mask_type='word'
        )

    def copywriting_generation(self,
                               text: str,
                               min_dec_len: int = 32,
                               seq_len: int = 512,
                               topp: float = 0.9,
                               penalty_score: float = 1.2) -> str:
        '''
        文案生成
        '''
        text = "标题：{} 文案：".format(text)
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='adtext',
            penalty_text='',
            choice_text='',
            is_unidirectional=False,
            min_dec_penalty_text='',
            logits_bias=-10000,
            mask_type='word'
        )

    def novel_continuation(self,
                           text: str,
                           min_dec_len: int = 2,
                           seq_len: int = 512,
                           topp: float = 0.9,
                           penalty_score: float = 1.2) -> str:
        '''
        小说续写
        '''
        text = "上文：{} 下文：".format(text)
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='gPARAGRAPH',
            penalty_text='',
            choice_text='',
            is_unidirectional=True,
            min_dec_penalty_text='。？：！[<S>]',
            logits_bias=-5,
            mask_type='paragraph'
        )

    def answer_generation(self,
                          text: str,
                          min_dec_len: int = 2,
                          seq_len: int = 512,
                          topp: float = 0.9,
                          penalty_score: float = 1.2) -> str:
        '''
        自由问答
        '''
        text = "问题：{} 回答：".format(text)
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='qa',
            penalty_text='[gEND]',
            choice_text='',
            is_unidirectional=True,
            min_dec_penalty_text='。？：！[<S>]',
            logits_bias=-5,
            mask_type='paragraph'
        )

    def couplet_continuation(self,
                             text: str,
                             min_dec_len: int = 2,
                             seq_len: int = 512,
                             topp: float = 0.9,
                             penalty_score: float = 1.0) -> str:
        '''
        对联续写
        '''
        text = "上联：{} 下联：".format(text)
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='couplet',
            penalty_text='',
            choice_text='',
            is_unidirectional=False,
            min_dec_penalty_text='',
            logits_bias=-10000,
            mask_type='word'
        )

    def composition_generation(self,
                               text: str,
                               min_dec_len: int = 128,
                               seq_len: int = 512,
                               topp: float = 0.9,
                               penalty_score: float = 1.2) -> str:
        '''
        作文创作
        '''
        text = "作文题目：{} 正文：".format(text)
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='zuowen',
            penalty_text='',
            choice_text='',
            is_unidirectional=False,
            min_dec_penalty_text='',
            logits_bias=-10000,
            mask_type='word'
        )

    def text_cloze(self,
                   text: str,
                   min_dec_len: int = 1,
                   seq_len: int = 512,
                   topp: float = 0.9,
                   penalty_score: float = 1.0) -> str:
        '''
        完形填空
        '''
        return self.custom_generation(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score,
            stop_token='',
            task_prompt='cloze',
            penalty_text='',
            choice_text='',
            is_unidirectional=False,
            min_dec_penalty_text='',
            logits_bias=-10000,
            mask_type='word'
        )

    @runnable
    def cmd(self, argvs):
        parser = argparse.ArgumentParser(
            description="Run the {}".format(self.name),
            prog="hub run {}".format(self.name),
            usage='%(prog)s',
            add_help=True)

        parser.add_argument('--text', type=str, required=True)
        parser.add_argument('--min_dec_len', type=int, default=1)
        parser.add_argument('--seq_len', type=int, default=128)
        parser.add_argument('--topp', type=float, default=1.0)
        parser.add_argument('--penalty_score', type=float, default=1.0)
        parser.add_argument('--stop_token', type=str, default='')
        parser.add_argument('--task_prompt', type=str, default='')
        parser.add_argument('--penalty_text', type=str, default='')
        parser.add_argument('--choice_text', type=str, default='')
        parser.add_argument('--is_unidirectional', type=bool, default=False)
        parser.add_argument('--min_dec_penalty_text', type=str, default='')
        parser.add_argument('--logits_bias', type=int, default=-10000)
        parser.add_argument('--mask_type', type=str, default='word')
        parser.add_argument('--ak', type=str, default='')
        parser.add_argument('--sk', type=str, default='')
        parser.add_argument('--task', type=str, default='custom_generation')

        args = parser.parse_args(argvs)

        func = getattr(self, args.task)

        if (args.ak != '') and (args.sk != ''):
            self.access_token = get_access_token(args.ak, args.sk)

        kwargs = vars(args)
        if kwargs['task'] not in ['custom_generation']:
            kwargs.pop('stop_token')
            kwargs.pop('task_prompt')
            kwargs.pop('penalty_text')
            kwargs.pop('choice_text')
            kwargs.pop('is_unidirectional')
            kwargs.pop('min_dec_penalty_text')
            kwargs.pop('logits_bias')
            kwargs.pop('mask_type')
            default_kwargs = {
                'min_dec_len': 1,
                'seq_len': 128,
                'topp': 1.0,
                'penalty_score': 1.0
            }
        else:
            default_kwargs = {
                'min_dec_len': 1,
                'seq_len': 128,
                'topp': 1.0,
                'penalty_score': 1.0,
                'stop_token': '',
                'task_prompt': '',
                'penalty_text': '',
                'choice_text': '',
                'is_unidirectional': False,
                'min_dec_penalty_text': '',
                'logits_bias': -10000,
                'mask_type': 'word'
            }
        kwargs.pop('task')
        kwargs.pop('ak')
        kwargs.pop('sk')

        for k in default_kwargs.keys():
            if kwargs[k] == default_kwargs[k]:
                kwargs.pop(k)

        return func(**kwargs)


if __name__ == '__main__':
    ernie_zeus = ERNIEZeus()

    result = ernie_zeus.custom_generation(
        '你好，'
    )
    print(result)

    result = ernie_zeus.text_generation(
        '给宠物猫起一些可爱的名字。名字：'
    )
    print(result)

    result = ernie_zeus.text_summarization(
        '在芬兰、瑞典提交“入约”申请近一个月来，北约成员国内部尚未对此达成一致意见。与此同时，俄罗斯方面也多次对北约“第六轮扩张”发出警告。据北约官网显示，北约秘书长斯托尔滕贝格将于本月12日至13日出访瑞典和芬兰，并将分别与两国领导人进行会晤。'
    )
    print(result)

    result = ernie_zeus.copywriting_generation(
        '芍药香氛的沐浴乳'
    )
    print(result)

    result = ernie_zeus.novel_continuation(
        '昆仑山可以说是天下龙脉的根源，所有的山脉都可以看作是昆仑的分支。这些分出来的枝枝杈杈，都可以看作是一条条独立的龙脉。'
    )
    print(result)

    result = ernie_zeus.answer_generation(
        '交朋友的原则是什么？'
    )
    print(result)

    result = ernie_zeus.couplet_continuation(
        '五湖四海皆春色'
    )
    print(result)

    result = ernie_zeus.composition_generation(
        '诚以养德，信以修身'
    )
    print(result)

    result = ernie_zeus.text_cloze(
        '她有着一双[MASK]的眼眸。'
    )
    print(result)
