# coding:utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import shutil
from copy import deepcopy

import numpy as np
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
try:
    from ernie.modeling_ernie import ErnieModelForGeneration
    from ernie.tokenizing_ernie import ErnieTokenizer
    from ernie.optimization import AdamW, LinearDecay
except:
    raise ImportError(
        "The module requires additional dependencies: ernie. You can install ernie via 'pip install paddle-ernie'"
    )
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo

from .decode import beam_search_infilling, post_process
import ernie_gen.propeller.paddle as propeller


@moduleinfo(
    name="ernie_gen",
    version="1.0.1",
    summary=
    "ERNIE-GEN is a multi-flow language generation framework for both pre-training and fine-tuning.",
    author="baidu",
    author_email="",
    type="nlp/text_generation",
)
class ErnieGen(hub.Module):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.tokenizer = ErnieTokenizer.from_pretrained(
            "ernie-1.0", mask_token=None)
        self.rev_dict = {v: k for k, v in self.tokenizer.vocab.items()}
        self.rev_lookup = np.vectorize(lambda i: self.rev_dict[i])
        self._model = None

    @property
    def model(self):
        if not self._model:
            self._model = ErnieModelForGeneration.from_pretrained("ernie-1.0")
        return self._model

    def finetune(
            self,
            train_path,
            dev_path=None,
            save_dir="ernie_gen_result",
            init_ckpt_path=None,
            use_gpu=True,
            max_steps=500,
            batch_size=8,
            max_encode_len=50,
            max_decode_len=50,
            learning_rate=5e-5,
            warmup_proportion=0.1,
            weight_decay=0.1,
            noise_prob=0,
            label_smooth=0,
            beam_width=5,
            length_penalty=1.0,
            log_interval=100,
            save_interval=200,
    ):
        """
        finetune with the specified dataset.

        Args:
            train_path(str): the train dataset path.
            dev_path(str): the dev dataset path.
            save_dir(str): the model params and dev dataset predict result save path.
            init_ckpt_path(str): incremental training load path.
            use_gpu(bool): use gpu or not.
            max_steps(int): max training steps.
            batch_size(int): the batch size.
            max_encode_len(int): the max encode length.
            max_decode_len(int): the max decode length.
            learning_rate(float): the learning rate.
            warmup_proportion(float): the warmup proportion.
            weight_decay(float): the weight decay magnitude.
            noise_prob(float): the nosie probability. see the ernie gen paper for details.
            label_smooth(float): the label smooth magnitude.
            beam_width(int): the beam size during evaluating the dev dataset.
            length_penalty(float): the length penalty during evaluating the dev dataset.
            log_interval(int): the log interval.
            save_interval(int): the save interval. dev set will be evaluated after saving.

        Return:
            result(dict): A Dictionary of shape::
                {
                    last_save_path(str): last model save path.
                    last_ppl(float): last model ppl.
                }
        """
        self.max_encode_len = max_encode_len
        self.max_decode_len = max_decode_len
        self.noise_prob = noise_prob

        place = F.CUDAPlace(0) if use_gpu else F.CPUPlace()

        with F.dygraph.guard(place):
            if init_ckpt_path is not None:
                logger.info('loading checkpoint from %s' % init_ckpt_path)
                sd, _ = D.load_dygraph(init_ckpt_path)
                self.model.set_dict(sd)

            feature_column = propeller.data.FeatureColumns([
                propeller.data.LabelColumn('id'),
                propeller.data.TextColumn(
                    'src',
                    unk_id=self.tokenizer.unk_id,
                    vocab_dict=self.tokenizer.vocab,
                    tokenizer=self.tokenizer.tokenize),
                propeller.data.TextColumn(
                    'tgt',
                    unk_id=self.tokenizer.unk_id,
                    vocab_dict=self.tokenizer.vocab,
                    tokenizer=self.tokenizer.tokenize),
            ])

            train_ds = feature_column.build_dataset('train', data_file=train_path, shuffle=False,
                                                    repeat=True, use_gz=False)\
                .map(self._map_fn).shuffle(10000).padded_batch(batch_size).map(self._after_padding)
            train_ds.data_shapes = [[None, None]] * 7 + [[None, None, None]
                                                         ] * 3 + [[None]]
            train_ds.data_types = ['int64'] * 11

            if dev_path:
                dev_ds = feature_column.build_dataset('dev', data_file=dev_path, shuffle=False,
                                                    repeat=False, use_gz=False) \
                    .map(self._map_fn) \
                    .padded_batch(1) \
                    .map(self._after_padding)
                dev_ds.data_shapes = [[None, None]] * 7 + [[None, None, None]
                                                           ] * 3 + [[None]]
                dev_ds.data_types = ['int64'] * 11

            vocab_size, _ = self.model.word_emb.weight.shape
            g_clip = F.clip.GradientClipByGlobalNorm(1.0)
            opt = AdamW(
                learning_rate=LinearDecay(learning_rate,
                                          int(warmup_proportion * max_steps),
                                          max_steps),
                parameter_list=self.model.parameters(),
                weight_decay=weight_decay,
                grad_clip=g_clip)

            loss = None

            save_path = None
            ppl = None

            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for step, data in enumerate(train_ds.start(place)):
                (example_id, src_ids, src_sids, src_pids, tgt_ids, tgt_sids,
                 tgt_pids, attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
                 mask_attn_2_srctgtattn, tgt_labels) = data

                _, __, info = self.model(
                    src_ids,
                    sent_ids=src_sids,
                    pos_ids=src_pids,
                    attn_bias=mask_src_2_src,
                    encode_only=True)
                cached_k, cached_v = info['caches']
                _, __, info = self.model(
                    tgt_ids,
                    sent_ids=tgt_sids,
                    pos_ids=tgt_pids,
                    attn_bias=mask_tgt_2_srctgt,
                    past_cache=(cached_k, cached_v),
                    encode_only=True)
                cached_k2, cached_v2 = info['caches']
                past_cache_k = [
                    L.concat([k, k2], 1) for k, k2 in zip(cached_k, cached_k2)
                ]
                past_cache_v = [
                    L.concat([v, v2], 1) for v, v2 in zip(cached_v, cached_v2)
                ]
                if label_smooth > 0.:
                    tgt_labels = L.label_smooth(
                        F.one_hot(tgt_labels, vocab_size), epsilon=label_smooth)
                loss, _, __ = self.model(
                    attn_ids,
                    sent_ids=tgt_sids,
                    pos_ids=tgt_pids,
                    attn_bias=mask_attn_2_srctgtattn,
                    past_cache=(past_cache_k, past_cache_v),
                    tgt_labels=tgt_labels,
                    tgt_pos=L.where(attn_ids == self.tokenizer.vocab['[MASK]']))

                loss.backward()
                opt.minimize(loss)
                self.model.clear_gradients()

                if step % log_interval == 0:
                    loss_np = loss.numpy()
                    ppl = np.exp(loss_np)
                    logger.info(
                        '[step %d / %d]train loss %.5f, ppl %.5f, elr %.3e' %
                        (step, max_steps, loss_np, ppl, opt.current_step_lr()))
                if save_dir and step % save_interval == 0 and step > 0:
                    loss_np = loss.numpy()
                    ppl = np.exp(loss_np)
                    save_name = "step_%s_ppl_%.5f" % (step, ppl)
                    save_path = os.path.join(save_dir, save_name)
                    logger.info("save the model in %s" % save_path)
                    F.save_dygraph(self.model.state_dict(), save_path)

                    if dev_path:
                        logger.info('evaluating...')
                        res = self._evaluate(dev_ds, place, beam_width,
                                             length_penalty)
                        output_path = os.path.join(
                            save_dir, "step_%s_ppl_%.5f.txt" % (step, ppl))
                        logger.info(
                            'save the predict result in %s' % output_path)
                        with open(output_path, 'w') as fout:
                            fout.write(('\n'.join(res)))

                if step > max_steps:
                    break

            if loss:
                loss_np = loss.numpy()
                ppl = np.exp(loss_np)
                logger.info('[final step %d]train loss %.5f, ppl %.5f, elr %.3e'
                            % (step, loss_np, ppl, opt.current_step_lr()))
                if save_dir:
                    save_name = "step_%s_ppl_%.5f" % (step, ppl)
                    save_path = os.path.join(save_dir, save_name)
                    logger.info("save the model in %s" % save_path)
                    F.save_dygraph(self.model.state_dict(), save_path)

                    if dev_path:
                        logger.info('evaluating...')
                        res = self._evaluate(dev_ds, place, beam_width,
                                             length_penalty)
                        output_path = os.path.join(
                            save_dir, "step_%s_ppl_%.5f.txt" % (step, ppl))
                        logger.info(
                            'save the predict result in %s' % output_path)
                        with open(output_path, 'w') as fout:
                            fout.write(('\n'.join(res)))

            result = {
                "last_save_path": "%s.pdparams" % save_path,
                "last_ppl": ppl[0],
            }

            return result

    def export(self,
               params_path,
               module_name,
               author,
               max_encode_len=50,
               max_decode_len=50,
               version="1.0.0",
               summary="",
               author_email="",
               export_path="."):
        """
        export the model saved in the params_path to a hub module.

        Args:
            params_path(str): the model params save path.
            module_name(str): the module name.
            author(str): the author name.
            max_encode_len(int): the max encode length.
            max_decode_len(int): the max decode length.
            version(str): the version information.
            summary(str): the module brief introduction.
            author_email(str): the author email address.
            export_path(str): the module export path.
        """
        if not os.path.exists(params_path):
            raise FileNotFoundError("The path %s does not exist." % params_path)
        export_module_path = os.path.join(export_path, module_name)
        if not os.path.exists(export_module_path):
            os.makedirs(export_module_path)
        logger.info("Begin export the model save in %s ..." % params_path)

        assets_path = os.path.join(self.directory, "template", "assets")
        model_path = os.path.join(self.directory, "template", "model")
        init_path = os.path.join(self.directory, "template", "__init__.py")
        module_temp_path = os.path.join(self.directory, "template",
                                        "module.temp")

        export_assets_path = os.path.join(export_module_path, "assets")
        export_params_path = os.path.join(export_module_path, "assets",
                                          "ernie_gen.pdparams")
        export_init_path = os.path.join(export_module_path, "__init__.py")
        export_model_path = os.path.join(export_module_path, "model")

        shutil.copyfile(init_path, export_init_path)
        shutil.copytree(assets_path, export_assets_path)
        shutil.copyfile(params_path, export_params_path)
        shutil.copytree(model_path, export_model_path)

        module_path = os.path.join(export_module_path, "module.py")
        with open(
                module_temp_path, encoding="utf8") as ftemp, open(
                    module_path, "w") as fmodule:
            content = ftemp.read().replace(
                r"{module_name}",
                module_name).replace(r"{author}", author).replace(
                    r"{version}",
                    version).replace(r"{summary}", summary).replace(
                        r"{author_email}", author_email).replace(
                            r"{max_encode_len}", str(max_encode_len)).replace(
                                r"{max_decode_len}", str(max_decode_len))
            fmodule.write(content)

        logger.info("The module has exported to %s" %
                    os.path.abspath(export_module_path))

    def _evaluate(self, datasets, place, beam_width, length_penalty):
        self.model.eval()
        printables = []
        for step, data in enumerate(datasets.start(place)):
            (example_id, src_ids, src_sids, src_pids, _, _, _, _, _, _, _,
             _) = data  # never use target when infer
            output_ids = beam_search_infilling(
                self.model,
                src_ids,
                src_sids,
                eos_id=self.tokenizer.sep_id,
                sos_id=self.tokenizer.cls_id,
                attn_id=self.tokenizer.vocab["[MASK]"],
                max_decode_len=self.max_decode_len,
                max_encode_len=self.max_encode_len,
                beam_width=beam_width,
                length_penalty=length_penalty,
                tgt_type_id=1,
            )
            output_str = self.rev_lookup(output_ids.numpy())
            for eid, ostr in zip(example_id.numpy().tolist(),
                                 output_str.tolist()):
                if '[SEP]' in ostr:
                    ostr = ostr[:ostr.index('[SEP]')]
                ostr = ''.join(map(post_process, ostr))
                printables.append('%d\t%s' % (eid, ostr))
        self.model.train()
        return printables

    def _map_fn(self, example_id, src_ids, tgt_ids):
        src_ids = src_ids[:self.max_encode_len]
        tgt_ids = tgt_ids[:self.max_decode_len]
        src_ids, src_sids = self.tokenizer.build_for_ernie(src_ids)
        src_pids = np.arange(len(src_ids), dtype=np.int64)

        tgt_ids, tgt_sids = self.tokenizer.build_for_ernie(tgt_ids)
        tgt_pids = np.arange(
            len(tgt_ids), dtype=np.int64) + len(src_ids)  # continues position
        tgt_sids = np.ones_like(tgt_sids)

        attn_ids = np.ones_like(tgt_ids) * self.tokenizer.vocab['[MASK]']
        if self.noise_prob > 0.:
            tgt_labels = deepcopy(tgt_ids)
            tgt_ids = self._make_some_noise(tgt_ids,
                                            self.noise_prob)  #corrupted
        else:
            tgt_labels = tgt_ids

        return (example_id, src_ids, src_pids, src_sids, tgt_ids, tgt_pids,
                tgt_sids, attn_ids, tgt_labels)

    def _make_some_noise(self, ids, noise_prob):
        noise_ids = np.random.randint(
            1, len(self.tokenizer.vocab), size=ids.shape)
        pos, = np.where(np.ones_like(ids))
        np.random.shuffle(pos)
        pos = pos[:int(noise_prob * len(pos))]
        ids[pos, ] = noise_ids[pos, ]
        return ids

    def _after_padding(self, example_id, src_ids, src_pids, src_sids, tgt_ids,
                       tgt_pids, tgt_sids, attn_ids, tgt_labels):
        '''
        attention mask:
        ***  src,  tgt, attn
        src  00,   01,   11
        tgt  10,   11,   12
        attn 20,   21,   22
        ***   s1, s2 | t1 t2 t3| attn1 attn2 attn3
        s1    1,  1  | 0, 0, 0,| 0,    0,    0,
        s2    1,  1  | 0, 0, 0,| 0,    0,    0,
        -
        t1    1,  1, | 1, 0, 0,| 0,    0,    0,
        t2    1,  1, | 1, 1, 0,| 0,    0,    0,
        t3    1,  1, | 1, 1, 1,| 0,    0,    0,
        -
        attn1 1,  1, | 0, 0, 0,| 1,    0,    0,
        attn2 1,  1, | 1, 0, 0,| 0,    1,    0,
        attn3 1,  1, | 1, 1, 0,| 0,    0,    1,
        for details, see Fig3. https://arxiv.org/abs/2001.11314
        '''

        src_len = src_ids.shape[1]
        tgt_len = tgt_ids.shape[1]
        mask_00 = self._gen_mask(src_ids, 'bidi', query_len=src_len)

        mask_10 = self._gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_11 = self._gen_mask(tgt_ids, 'causal', query_len=tgt_len)

        mask_20 = self._gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_21 = self._gen_mask(
            tgt_ids, 'causal_without_diag', query_len=tgt_len)
        mask_22 = self._gen_mask(attn_ids, 'diag', query_len=tgt_len)
        '''
        mask = np.concatenate([
            np.concatenate([mask_00, mask_01, mask_02], 2),
            np.concatenate([mask_10, mask_11, mask_12], 2),
            np.concatenate([mask_20, mask_21, mask_22], 2),
        ], 1)
        ids = np.concatenate([src_ids, tgt_ids, attn_ids], 1)
        pids = np.concatenate([src_pids, tgt_pids, tgt_pids], 1)
        sids = np.concatenate([src_sids, tgt_sids, tgt_sids], 1)
        '''

        mask_src_2_src = mask_00
        mask_tgt_2_srctgt = np.concatenate([mask_10, mask_11], 2)
        mask_attn_2_srctgtattn = np.concatenate([mask_20, mask_21, mask_22], 2)

        tgt_labels = tgt_labels[np.where(tgt_labels != 0)]
        return (example_id, src_ids, src_sids, src_pids, tgt_ids, tgt_sids,
                tgt_pids, attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
                mask_attn_2_srctgtattn, tgt_labels)

    def _gen_mask(self,
                  batch_ids,
                  mask_type='bidi',
                  query_len=None,
                  pad_value=0):
        if query_len is None:
            query_len = batch_ids.shape[1]
        if mask_type != 'empty':
            mask = (batch_ids != pad_value).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
            if mask_type == 'causal':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask)
            elif mask_type == 'causal_without_diag':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask, -1)
            elif mask_type == 'diag':
                assert query_len == batch_ids.shape[1]
                mask = np.stack([np.diag(np.diag(m)) for m in mask], 0)
        else:
            mask = np.zeros_like(batch_ids).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
        return mask


if __name__ == "__main__":
    module = ErnieGen()
    result = module.finetune(
        train_path='test_data/train.txt',
        dev_path='test_data/dev.txt',
        max_steps=300,
        batch_size=2)
    module.export(
        params_path=result['last_save_path'],
        module_name="ernie_gen_test",
        author="test")
