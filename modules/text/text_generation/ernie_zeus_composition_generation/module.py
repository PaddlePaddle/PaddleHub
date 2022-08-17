import argparse

import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable


@moduleinfo(
    name='ernie_zeus_composition_generation',
    type='nlp/text_generation',
    author='paddlepaddle',
    author_email='',
    summary='ernie_zeus_composition_generation',
    version='1.0.0'
)
class ERNIEZeus:
    def __init__(self) -> None:
        self.ernie_zeus = hub.Module(name='ernie_zeus')

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
        return self.ernie_zeus.composition_generation(
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

        return self.composition_generation(**kwargs)


if __name__ == '__main__':
    ernie_zeus = ERNIEZeus()

    result = ernie_zeus.composition_generation(
        '诚以养德，信以修身'
    )
    print(result)
