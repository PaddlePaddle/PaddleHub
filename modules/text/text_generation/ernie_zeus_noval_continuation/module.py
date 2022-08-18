import argparse

import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable


@moduleinfo(
    name='ernie_zeus_novel_continuation',
    type='nlp/text_generation',
    author='paddlepaddle',
    author_email='',
    summary='ernie_zeus_novel_continuation',
    version='1.0.0'
)
class ERNIEZeus:
    def __init__(self) -> None:
        self.ernie_zeus = hub.Module(name='ernie_zeus')

    def novel_continuation(self,
                           text: str,
                           min_dec_len: int = 2,
                           seq_len: int = 512,
                           topp: float = 0.9,
                           penalty_score: float = 1.2) -> str:
        '''
        小说续写
        '''
        return self.ernie_zeus.novel_continuation(
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

        return self.novel_continuation(**kwargs)


if __name__ == '__main__':
    ernie_zeus = ERNIEZeus()

    result = ernie_zeus.novel_continuation(
        '昆仑山可以说是天下龙脉的根源，所有的山脉都可以看作是昆仑的分支。这些分出来的枝枝杈杈，都可以看作是一条条独立的龙脉。'
    )
    print(result)
