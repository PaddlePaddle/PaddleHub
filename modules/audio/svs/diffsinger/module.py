import argparse
import os
import time
from typing import Dict
from typing import List
from typing import Union

from .infer import Infer
from .utils.audio import save_wav
from .utils.hparams import hparams
from .utils.hparams import set_hparams
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="diffsinger",
            type="Audio/svs",
            author="",
            author_email="",
            summary="DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism",
            version="1.0.0")
class DiffSinger:

    def __init__(self, providers: List[str] = None) -> None:
        root = self.directory
        config = os.path.join('model', 'config.yaml')
        set_hparams(config, root=root)
        self.infer = Infer(root, providers=providers)

    @serving
    def singing_voice_synthesis(self,
                                inputs: Dict[str, str],
                                sample_num: int = 1,
                                save_audio: bool = True,
                                save_dir: str = 'outputs') -> Dict[str, Union[List[List[int]], int]]:
        '''
        inputs = {
            'text': '小酒窝长睫毛AP是你最美的记号',
            'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
            'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
            'input_type': 'word'
        }  # user input: Chinese characters
        or,
        inputs = {
            'text': '小酒窝长睫毛AP是你最美的记号',
            'ph_seq': 'x iao j iu w o ch ang ang j ie ie m ao AP sh i n i z ui m ei d e j i h ao',
            'note_seq': 'C#4/Db4 C#4/Db4 F#4/Gb4 F#4/Gb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 F#4/Gb4 F#4/Gb4 F#4/Gb4 C#4/Db4 C#4/Db4 C#4/Db4 rest C#4/Db4 C#4/Db4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 F4 F4 C#4/Db4 C#4/Db4',
            'note_dur_seq': '0.407140 0.407140 0.376190 0.376190 0.242180 0.242180 0.509550 0.509550 0.183420 0.315400 0.315400 0.235020 0.361660 0.361660 0.223070 0.377270 0.377270 0.340550 0.340550 0.299620 0.299620 0.344510 0.344510 0.283770 0.283770 0.323390 0.323390 0.360340 0.360340',
            'is_slur_seq': '0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
            'input_type': 'phoneme'
        }  # input like Opencpop dataset.
        '''
        outputs = []
        for i in range(sample_num):
            output = self.infer.infer_once(inputs)
            os.makedirs(save_dir, exist_ok=True)
            if save_audio:
                save_wav(output, os.path.join(save_dir, '%d_%d.wav' % (i, int(time.time()))),
                         hparams['audio_sample_rate'])
            outputs.append(output.tolist())
        return {'wavs': outputs, 'sample_rate': hparams['audio_sample_rate']}

    @runnable
    def run_cmd(self, argvs: List[str]) -> str:
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.parser.add_argument('--input_type',
                                 type=str,
                                 choices=['word', 'phoneme'],
                                 required=True,
                                 help='input type in ["word", "phoneme"].')
        args = self.parser.parse_args(argvs[:2])
        if args.input_type == 'word':
            self.arg_input_group = self.parser.add_argument_group(title="Input options (type: word).",
                                                                  description="Input options (type: word).")
            self.arg_input_group.add_argument('--text', type=str, required=True, help='input text.')
            self.arg_input_group.add_argument('--notes', type=str, required=True, help='input notes.')
            self.arg_input_group.add_argument('--notes_duration', type=str, required=True, help='input notes duration.')
        elif args.input_type == 'phoneme':
            self.arg_input_group = self.parser.add_argument_group(title="Input options (type: phoneme).",
                                                                  description="Input options (type: phoneme).")
            self.arg_input_group.add_argument('--text', type=str, required=True, help='input text.')
            self.arg_input_group.add_argument('--ph_seq', type=str, required=True, help='input phoneme seq.')
            self.arg_input_group.add_argument('--note_seq', type=str, required=True, help='input note seq.')
            self.arg_input_group.add_argument('--note_dur_seq',
                                              type=str,
                                              required=True,
                                              help='input note duration seq.')
            self.arg_input_group.add_argument('--is_slur_seq',
                                              type=str,
                                              required=True,
                                              help='input if note is slur seq.')
        else:
            raise ValueError('Input type (--input_type) should be in ["word", "phoneme"]')
        self.parser.add_argument('--sample_num', type=int, default=1, help='sample audios num, default=1')
        self.parser.add_argument('--save_dir',
                                 type=str,
                                 default='outputs',
                                 help='sample audios save_dir, default="outputs"')
        args = self.parser.parse_args(argvs)
        kwargs = vars(args).copy()
        kwargs.pop('sample_num')
        kwargs.pop('save_dir')
        self.singing_voice_synthesis(kwargs, sample_num=args.sample_num, save_dir=args.save_dir, save_audio=True)
        return "Audios are saved in %s" % args.save_dir
