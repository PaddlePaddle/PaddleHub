import shutil
import unittest

import paddlehub as hub


class TestHubModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = hub.Module(name="diffsinger")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('outputs')

    def test_singing_voice_synthesis1(self):
        results = self.module.singing_voice_synthesis(inputs={
            'text': '小酒窝长睫毛AP是你最美的记号',
            'notes':
            'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
            'notes_duration':
            '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340',
            'input_type': 'word'
        },
                                                      sample_num=1,
                                                      save_audio=True,
                                                      save_dir='outputs')
        self.assertIsInstance(results, dict)
        self.assertIsInstance(results['wavs'], list)
        self.assertIsInstance(results['wavs'][0], list)
        self.assertEqual(len(results['wavs'][0]), 123776)
        self.assertEqual(results['sample_rate'], 24000)

    def test_singing_voice_synthesis2(self):
        results = self.module.singing_voice_synthesis(inputs={
            'text': '小酒窝长睫毛AP是你最美的记号',
            'ph_seq': 'x iao j iu w o ch ang ang j ie ie m ao AP sh i n i z ui m ei d e j i h ao',
            'note_seq':
            'C#4/Db4 C#4/Db4 F#4/Gb4 F#4/Gb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 F#4/Gb4 F#4/Gb4 F#4/Gb4 C#4/Db4 C#4/Db4 C#4/Db4 rest C#4/Db4 C#4/Db4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 F4 F4 C#4/Db4 C#4/Db4',
            'note_dur_seq':
            '0.407140 0.407140 0.376190 0.376190 0.242180 0.242180 0.509550 0.509550 0.183420 0.315400 0.315400 0.235020 0.361660 0.361660 0.223070 0.377270 0.377270 0.340550 0.340550 0.299620 0.299620 0.344510 0.344510 0.283770 0.283770 0.323390 0.323390 0.360340 0.360340',
            'is_slur_seq': '0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
            'input_type': 'phoneme'
        },
                                                      sample_num=1,
                                                      save_audio=True,
                                                      save_dir='outputs')
        self.assertIsInstance(results, dict)
        self.assertIsInstance(results['wavs'], list)
        self.assertIsInstance(results['wavs'][0], list)
        self.assertEqual(len(results['wavs'][0]), 123776)
        self.assertEqual(results['sample_rate'], 24000)


if __name__ == "__main__":
    unittest.main()
