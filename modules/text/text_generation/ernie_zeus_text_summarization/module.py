import argparse

import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable


@moduleinfo(
    name='ernie_zeus_text_summarization',
    type='nlp/text_generation',
    author='paddlepaddle',
    author_email='',
    summary='ernie_zeus_text_summarization',
    version='1.0.0'
)
class ERNIEZeus:
    def __init__(self) -> None:
        self.ernie_zeus = hub.Module(name='ernie_zeus')

    def text_summarization(self,
                           text: str,
                           min_dec_len: int = 4,
                           seq_len: int = 512,
                           topp: float = 0.0,
                           penalty_score: float = 1.0) -> str:
        '''
        摘要生成
        '''
        return self.ernie_zeus.text_summarization(
            text,
            min_dec_len,
            seq_len,
            topp,
            penalty_score
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

        default_kwargs = {
            'min_dec_len': 1,
            'seq_len': 128,
            'topp': 1.0,
            'penalty_score': 1.0,
        }

        args = parser.parse_args(argvs)

        kwargs = vars(args)

        for k in default_kwargs.keys():
            if kwargs[k] == default_kwargs[k]:
                kwargs.pop(k)

        return self.text_summarization(**kwargs)


if __name__ == '__main__':
    ernie_zeus = ERNIEZeus()

    result = ernie_zeus.text_summarization(
        '在芬兰、瑞典提交“入约”申请近一个月来，北约成员国内部尚未对此达成一致意见。与此同时，俄罗斯方面也多次对北约“第六轮扩张”发出警告。据北约官网显示，北约秘书长斯托尔滕贝格将于本月12日至13日出访瑞典和芬兰，并将分别与两国领导人进行会晤。'
    )
    print(result)
