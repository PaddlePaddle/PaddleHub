import os
from collections import deque

import librosa
import numpy as np
import onnxruntime as rt
from pypinyin import lazy_pinyin
from tqdm import tqdm

from .inference.svs.opencpop.map import cpop_pinyin2ph_func
from .utils.hparams import hparams
from .utils.text_encoder import TokenTextEncoder


class Infer:

    def __init__(self, root='.', providers=None):
        model_dir = os.path.join(root, 'model')
        if providers is None:
            providers = rt.get_available_providers()
        print('Using these as onnxruntime providers:', providers)

        phone_list = [
            "AP", "SP", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f", "g", "h",
            "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "j", "k", "l", "m", "n", "o", "ong", "ou",
            "p", "q", "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "v", "van", "ve", "vn",
            "w", "x", "y", "z", "zh"
        ]
        self.ph_encoder = TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')
        self.pinyin2phs = cpop_pinyin2ph_func(path=os.path.join(root, 'inference/svs/opencpop/cpop_pinyin2ph.txt'))
        self.spk_map = {'opencpop': 0}

        options = rt.SessionOptions()
        for provider in providers:
            if 'dml' in provider.lower():
                options.enable_mem_pattern = False
                options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        fs2_path = os.path.join(model_dir, 'fs2.onnx')
        q_sample_path = os.path.join(model_dir, 'q_sample.onnx')
        p_sample_path = os.path.join(model_dir, 'p_sample.onnx')
        pe_path = os.path.join(model_dir, 'pe.onnx')
        vocoder_path = os.path.join(model_dir, 'vocoder.onnx')
        self.fs2 = rt.InferenceSession(fs2_path, options, providers=providers)
        self.q_sample = rt.InferenceSession(q_sample_path, options, providers=providers)
        self.p_sample = rt.InferenceSession(p_sample_path, options, providers=providers)
        self.pe = rt.InferenceSession(pe_path, options, providers=providers)
        self.vocoder = rt.InferenceSession(vocoder_path, options, providers=providers)

        self.K_step = hparams['K_step']
        self.spec_min = np.asarray(hparams['spec_min'], np.float32)[None, None, :hparams['keep_bins']]
        self.spec_max = np.asarray(hparams['spec_max'], np.float32)[None, None, :hparams['keep_bins']]
        self.mel_bins = hparams['audio_num_mel_bins']
        self.use_pe = hparams.get('pe_enable') is not None and hparams['pe_enable']

    def model(self, txt_tokens, **kwargs):
        fs_input_names = [node.name for node in self.fs2.get_inputs()]
        inputs = {'txt_tokens': txt_tokens}
        inputs.update({k: v for k, v in kwargs.items() if isinstance(v, np.ndarray) and k in fs_input_names})

        io_binding = self.fs2.io_binding()
        for k, v in inputs.items():
            io_binding.bind_cpu_input(k, v)
        io_binding.bind_output('decoder_inp')
        io_binding.bind_output('mel_out')
        if not self.use_pe:
            io_binding.bind_output('f0_denorm')
        self.fs2.run_with_iobinding(io_binding)
        decoder_inp, mel_out = io_binding.get_outputs()[:2]
        self.device_name = mel_out.device_name()
        ret = {'decoder_inp': decoder_inp, 'mel_out': mel_out}
        if not self.use_pe:
            ret.update({'f0_denorm': io_binding.get_outputs()[-1]})
        cond = decoder_inp.numpy().transpose([0, 2, 1])

        ret['fs2_mel'] = ret['mel_out']
        fs2_mels = mel_out.numpy()
        t = self.K_step
        fs2_mels = self.norm_spec(fs2_mels)
        fs2_mels = fs2_mels.transpose([0, 2, 1])[:, None, :, :]

        io_binding = self.q_sample.io_binding()
        io_binding.bind_cpu_input('x_start', fs2_mels)
        io_binding.bind_cpu_input('noise', np.random.randn(*fs2_mels.shape).astype(fs2_mels.dtype))
        io_binding.bind_cpu_input('t', np.asarray([t - 1], dtype=np.int64))
        io_binding.bind_output('x_next')
        self.q_sample.run_with_iobinding(io_binding)
        x = io_binding.get_outputs()[0].numpy()
        if hparams.get('gaussian_start') is not None and hparams['gaussian_start']:
            print('===> gaussion start.')
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = np.random.randn(*shape).astype(fs2_mels.dtype)

        cond = rt.OrtValue.ortvalue_from_numpy(cond, mel_out.device_name(), 0)
        x = rt.OrtValue.ortvalue_from_numpy(x, mel_out.device_name(), 0)

        if hparams.get('pndm_speedup'):
            self.noise_list = deque(maxlen=4)
            iteration_interval = hparams['pndm_speedup']
            interval = rt.OrtValue.ortvalue_from_numpy(np.asarray([iteration_interval], np.int64),
                                                       mel_out.device_name(), 0)
            for i in tqdm(reversed(range(0, t, iteration_interval)),
                          desc='sample time step',
                          total=t // iteration_interval):
                io_binding = self.p_sample_plms.io_binding()
                io_binding.bind_ortvalue_input('x', x)
                io_binding.bind_cpu_input('noise', np.random.randn(*x.shape).astype(x.dtype))
                io_binding.bind_ortvalue_input('cond', cond)
                io_binding.bind_cpu_input('t', np.asarray([i], dtype=np.int64))  # torch i-1 but here i
                io_binding.bind_ortvalue_input('interval', interval)
                io_binding.bind_output('x_next')
                self.p_sample_plms.run_with_iobinding(io_binding)
                x = io_binding.get_outputs()[0]
        else:
            for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                io_binding = self.p_sample.io_binding()
                io_binding.bind_ortvalue_input('x', x)
                io_binding.bind_cpu_input('noise', np.random.randn(*x.shape()).astype(np.float32))
                io_binding.bind_ortvalue_input('cond', cond)
                io_binding.bind_cpu_input('t', np.asarray([i], dtype=np.int64))  # torch i-1 but here i
                io_binding.bind_output('x_next')
                self.p_sample.run_with_iobinding(io_binding)
                x = io_binding.get_outputs()[0]
        x = x.numpy()[:, 0].transpose([0, 2, 1])
        mel2ph = kwargs.get('mel2ph', None)
        if mel2ph is not None:  # for singing
            ret['mel_out'] = self.denorm_spec(x) * ((mel2ph > 0).astype(np.float32)[:, :, None])
        else:
            ret['mel_out'] = self.denorm_spec(x)
        return ret

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')

        output = self.model(txt_tokens,
                            spk_id=spk_id,
                            ref_mels=None,
                            infer=True,
                            pitch_midi=sample['pitch_midi'],
                            midi_dur=sample['midi_dur'],
                            is_slur=sample['is_slur'])
        mel_out = output['mel_out']  # [B, T,80]
        mel_out = rt.OrtValue.ortvalue_from_numpy(mel_out, self.device_name, 0)
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            # pe predict from Pred mel
            io_binding = self.pe.io_binding()
            io_binding.bind_ortvalue_input('mel_input', mel_out)
            io_binding.bind_output('f0_denorm_pred')
            self.pe.run_with_iobinding(io_binding)
            f0_pred = io_binding.get_outputs()[0]
        else:
            f0_pred = output['f0_denorm']
        wav_out = self.run_vocoder(mel_out, f0=f0_pred.numpy())

        return wav_out[0]

    def run_vocoder(self, c, **kwargs):
        # c = c.transpose([0, 2, 1])  # [B, 80, T]
        f0 = kwargs.get('f0')  # [B, T]
        if f0 is not None and hparams.get('use_nsf'):
            y = self.vocoder.run(['wav_out'], {
                'mel_out': c,
                'f0': f0,
            })[0]  # .reshape([-1])
        else:
            y = self.vocoder.run(['wav_out'], {
                'mel_out': c,
            })[0]  # .reshape([-1])
            # [T]
        return y  # [None]

    def preprocess_word_level_input(self, inp):
        # Pypinyin can't solve polyphonic words
        text_raw = inp['text'].replace('最长', '最常').replace('长睫毛', '常睫毛') \
            .replace('那么长', '那么常').replace('多长', '多常') \
            .replace('很长', '很常')  # We hope someone could provide a better g2p module for us by opening pull requests.

        # lyric
        pinyins = lazy_pinyin(text_raw, strict=False)
        ph_per_word_lst = [self.pinyin2phs[pinyin.strip()] for pinyin in pinyins if pinyin.strip() in self.pinyin2phs]

        # Note
        note_per_word_lst = [x.strip() for x in inp['notes'].split('|') if x.strip() != '']
        mididur_per_word_lst = [x.strip() for x in inp['notes_duration'].split('|') if x.strip() != '']

        if len(note_per_word_lst) == len(ph_per_word_lst) == len(mididur_per_word_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            print(ph_per_word_lst, note_per_word_lst, mididur_per_word_lst)
            print(len(ph_per_word_lst), len(note_per_word_lst), len(mididur_per_word_lst))
            return None

        note_lst = []
        ph_lst = []
        midi_dur_lst = []
        is_slur = []
        for idx, ph_per_word in enumerate(ph_per_word_lst):
            # for phs in one word:
            # single ph like ['ai']  or multiple phs like ['n', 'i']
            ph_in_this_word = ph_per_word.split()

            # for notes in one word:
            # single note like ['D4'] or multiple notes like ['D4', 'E4'] which means a 'slur' here.
            note_in_this_word = note_per_word_lst[idx].split()
            midi_dur_in_this_word = mididur_per_word_lst[idx].split()
            # process for the model input
            # Step 1.
            #  Deal with note of 'not slur' case or the first note of 'slur' case
            #  j        ie
            #  F#4/Gb4  F#4/Gb4
            #  0        0
            for ph in ph_in_this_word:
                ph_lst.append(ph)
                note_lst.append(note_in_this_word[0])
                midi_dur_lst.append(midi_dur_in_this_word[0])
                is_slur.append(0)
            # step 2.
            #  Deal with the 2nd, 3rd... notes of 'slur' case
            #  j        ie         ie
            #  F#4/Gb4  F#4/Gb4    C#4/Db4
            #  0        0          1
            # is_slur = True, we should repeat the YUNMU to match the 2nd, 3rd... notes.
            if len(note_in_this_word) > 1:
                for idx in range(1, len(note_in_this_word)):
                    ph_lst.append(ph_in_this_word[-1])
                    note_lst.append(note_in_this_word[idx])
                    midi_dur_lst.append(midi_dur_in_this_word[idx])
                    is_slur.append(1)
        ph_seq = ' '.join(ph_lst)

        if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
            print(len(ph_lst), len(note_lst), len(midi_dur_lst))
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_phoneme_level_input(self, inp):
        ph_seq = inp['ph_seq']
        note_lst = inp['note_seq'].split()
        midi_dur_lst = inp['note_dur_seq'].split()
        is_slur = [float(x) for x in inp['is_slur_seq'].split()]
        print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
        if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_input(self, inp, input_type='word'):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', 'opencpop')

        # single spk
        spk_id = self.spk_map[spk_name]

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            ret = self.preprocess_word_level_input(inp)
        # like transcriptions.txt in Opencpop dataset.
        elif input_type == 'phoneme':
            ret = self.preprocess_phoneme_level_input(inp)
        else:
            print('Invalid input type.')
            return None

        if ret:
            ph_seq, note_lst, midi_dur_lst, is_slur = ret
        else:
            print('==========> Preprocess_word_level or phone_level input wrong.')
            return None

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0 for x in note_lst]
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print('Invalid Input Type.')
            return None

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {
            'item_name': item_name,
            'text': inp['text'],
            'ph': ph_seq,
            'spk_id': spk_id,
            'ph_token': ph_token,
            'pitch_midi': np.asarray(midis),
            'midi_dur': np.asarray(midi_dur_lst),
            'is_slur': np.asarray(is_slur),
        }
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = np.int64(item['ph_token'])[None, :]
        txt_lengths = np.int64([txt_tokens.shape[1]])
        spk_ids = np.asarray(item['spk_id'], np.int64)[None]

        pitch_midi = np.int64(item['pitch_midi'])[None, :hparams['max_frames']]
        midi_dur = np.float32(item['midi_dur'])[None, :hparams['max_frames']]
        is_slur = np.int64(item['is_slur'])[None, :hparams['max_frames']]

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur
        }
        return batch

    def infer_once(self, inp):
        inp = self.preprocess_input(inp, input_type=inp['input_type'] if inp.get('input_type') else 'word')
        output = self.forward_model(inp)
        return output
