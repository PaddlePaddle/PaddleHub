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
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Rouge1, Rouge2
from paddlenlp.transformers import ErnieTokenizer, ErnieForGeneration, LinearDecayWithWarmup

from .encode import convert_example, after_padding
from .decode import post_process, beam_search_infilling
from .model import StackModel


@moduleinfo(
    name="ernie_gen",
    version="1.1.0",
    summary="ERNIE-GEN is a multi-flow language generation framework for both pre-training and fine-tuning.",
    author="baidu",
    author_email="",
    type="nlp/text_generation",
)
class ErnieGen():
    def __init__(self):
        """
        initialize with the necessary elements
        """
        self.tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
        self.rev_dict = self.tokenizer.vocab.idx_to_token
        self.rev_lookup = np.vectorize(lambda i: self.rev_dict[i])
        self._model = None

    @property
    def model(self):
        if not self._model:
            self._model = ErnieForGeneration.from_pretrained("ernie-1.0")
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
        paddle.disable_static()
        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        if init_ckpt_path is not None:
            logger.info('loading checkpoint from %s' % init_ckpt_path)
            sd = paddle.load(init_ckpt_path)
            self.model.set_state_dict(sd)

        train_dataset = self._load_dataset(train_path)
        attn_id = self.tokenizer.vocab['[MASK]']
        trans_func = convert_example(tokenizer=self.tokenizer,
                                     attn_id=attn_id,
                                     tgt_type_id=1,
                                     max_encode_len=max_encode_len,
                                     max_decode_len=max_decode_len,
                                     noise_prob=noise_prob)

        train_dataset = train_dataset.map(trans_func)
        train_batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # src_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # src_pids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # src_tids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # tgt_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # tgt_pids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id),  # tgt_tids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # attn_ids
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id),  # tgt_labels
        ): after_padding(fn(samples))
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_sampler=train_batch_sampler,
                                       collate_fn=batchify_fn,
                                       num_workers=0,
                                       return_list=True)

        if dev_path:
            dev_dataset = self._load_dataset(dev_path)
            dev_dataset = dev_dataset.map(trans_func)
            dev_data_loader = DataLoader(dataset=dev_dataset,
                                         batch_size=batch_size,
                                         collate_fn=batchify_fn,
                                         num_workers=0,
                                         return_list=True)

        label_num = self.model.word_emb.weight.shape[0]
        train_model = StackModel(self.model)
        lr_scheduler = LinearDecayWithWarmup(learning_rate, max_steps, warmup_proportion)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
        optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler,
                                           parameters=self.model.parameters(),
                                           weight_decay=weight_decay,
                                           grad_clip=nn.ClipGradByGlobalNorm(1.0),
                                           apply_decay_param_fun=lambda x: x in decay_params)

        rouge1 = Rouge1()
        rouge2 = Rouge2()
        global_step = 1
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        while True:
            for batch in train_data_loader:
                (src_ids, src_tids, src_pids, tgt_ids, tgt_tids, tgt_pids, attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
                 mask_attn_2_srctgtattn, tgt_labels, _) = batch
                if label_smooth > 0.:
                    tgt_labels = nn.functional.label_smooth(nn.functional.one_hot(tgt_labels, label_num),
                                                            epsilon=label_smooth)

                tgt_pos = paddle.nonzero(attn_ids == attn_id)
                loss = train_model(src_ids, src_tids, src_pids, tgt_ids, tgt_tids, tgt_pids, attn_ids, mask_src_2_src,
                                   mask_tgt_2_srctgt, mask_attn_2_srctgtattn, tgt_labels, tgt_pos)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % log_interval == 0 and paddle.distributed.get_rank() == 0:
                    loss_np = loss.numpy()
                    ppl = np.exp(loss_np)
                    logger.info('[step %d / %d]train loss %.5f, ppl %.5f, elr %.3e' %
                                (global_step, max_steps, loss_np, ppl, lr_scheduler.get_lr()))
                if save_dir and global_step % save_interval == 0 and global_step > 0:
                    loss_np = loss.numpy()
                    ppl = np.exp(loss_np)
                    save_name = "step_%s_ppl_%.5f.params" % (global_step, ppl)
                    save_path = os.path.join(save_dir, save_name)
                    logger.info("save the model in %s" % save_path)
                    paddle.save(self.model.state_dict(), save_path)

                    if dev_path:
                        self._evaluate(self.model, dev_data_loader, self.tokenizer, rouge1, rouge2, attn_id,
                                       max_decode_len, max_encode_len, beam_width, length_penalty)

                if global_step >= max_steps:
                    break
                global_step += 1

            if global_step >= max_steps:
                break

        if global_step % save_interval != 0:
            loss_np = loss.numpy()
            ppl = np.exp(loss_np)
            logger.info('[final step %d]train loss %.5f, ppl %.5f, elr %.3e' %
                        (global_step, loss_np, ppl, lr_scheduler.get_lr()))
            if save_dir:
                save_name = "step_%s_ppl_%.5f.pdparams" % (global_step, ppl)
                save_path = os.path.join(save_dir, save_name)
                logger.info("save the model in %s" % save_path)
                paddle.save(self.model.state_dict(), save_path)

                if dev_path:
                    self._evaluate(self.model, dev_data_loader, self.tokenizer, rouge1, rouge2, attn_id, max_decode_len,
                                   max_encode_len, beam_width, length_penalty)

        result = {
            "last_save_path": "%s" % save_path,
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
        init_path = os.path.join(self.directory, "template", "__init__.py")
        decode_path = os.path.join(self.directory, "template", "decode.py")
        module_temp_path = os.path.join(self.directory, "template", "module.temp")

        export_assets_path = os.path.join(export_module_path, "assets")
        export_params_path = os.path.join(export_module_path, "assets", "ernie_gen.pdparams")
        export_init_path = os.path.join(export_module_path, "__init__.py")
        export_decode_path = os.path.join(export_module_path, "decode.py")

        if not os.path.exists(export_assets_path):
            os.makedirs(export_assets_path)
        shutil.copyfile(init_path, export_init_path)
        shutil.copyfile(params_path, export_params_path)
        shutil.copyfile(decode_path, export_decode_path)

        module_path = os.path.join(export_module_path, "module.py")
        with open(module_temp_path, encoding="utf8") as ftemp, open(module_path, "w") as fmodule:
            content = ftemp.read().replace(r"{module_name}", module_name).replace(r"{author}", author).replace(
                r"{version}", version).replace(r"{summary}", summary).replace(r"{author_email}", author_email).replace(
                    r"{max_encode_len}", str(max_encode_len)).replace(r"{max_decode_len}", str(max_decode_len))
            fmodule.write(content)

        logger.info("The module has exported to %s" % os.path.abspath(export_module_path))

    def _evaluate(self, model, data_loader, tokenizer, rouge1, rouge2, attn_id, max_decode_len, max_encode_len,
                  beam_width, length_penalty):
        paddle.disable_static()
        model.eval()

        vocab = tokenizer.vocab
        eos_id = vocab[tokenizer.sep_token]
        sos_id = vocab[tokenizer.cls_token]
        pad_id = vocab[tokenizer.pad_token]
        unk_id = vocab[tokenizer.unk_token]
        vocab_size = len(vocab)
        evaluated_sentences_ids = []
        reference_sentences_ids = []
        logger.info("Evaluating...")
        for data in data_loader:
            (src_ids, src_tids, src_pids, _, _, _, _, _, _, _, _, raw_tgt_labels) = data  # never use target when infer
            # Use greedy_search_infilling or beam_search_infilling to get predictions
            output_ids = beam_search_infilling(model,
                                               src_ids,
                                               src_tids,
                                               eos_id=eos_id,
                                               sos_id=sos_id,
                                               attn_id=attn_id,
                                               pad_id=pad_id,
                                               unk_id=unk_id,
                                               vocab_size=vocab_size,
                                               max_decode_len=max_decode_len,
                                               max_encode_len=max_encode_len,
                                               beam_width=beam_width,
                                               length_penalty=length_penalty,
                                               tgt_type_id=1)

            for ids in output_ids.tolist():
                if eos_id in ids:
                    ids = ids[:ids.index(eos_id)]
                evaluated_sentences_ids.append(ids[0])

            for ids in raw_tgt_labels.numpy().tolist():
                ids = ids[:ids.index(eos_id)]
                reference_sentences_ids.append(ids)

        score1 = rouge1.score(evaluated_sentences_ids, reference_sentences_ids)
        score2 = rouge2.score(evaluated_sentences_ids, reference_sentences_ids)

        logger.info("Rouge-1: %.5f ,Rouge-2: %.5f" % (score1 * 100, score2 * 100))

        evaluated_sentences = []
        reference_sentences = []
        for ids in reference_sentences_ids[:3]:
            reference_sentences.append(''.join(map(post_process, vocab.to_tokens(ids))))
        for ids in evaluated_sentences_ids[:3]:
            evaluated_sentences.append(''.join(map(post_process, vocab.to_tokens(ids))))
        logger.debug(reference_sentences)
        logger.debug(evaluated_sentences)

        model.train()

    def _load_dataset(self, datafiles):
        def read(data_path):
            with open(data_path, 'r', encoding='utf-8') as fp:
                for line in fp.readlines():
                    order, words, labels = line.strip('\n').split('\t')
                    yield {'tokens': words, 'labels': labels}

        if isinstance(datafiles, str):
            return MapDataset(list(read(datafiles)))
        elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
            return [MapDataset(list(read(datafile))) for datafile in datafiles]


if __name__ == "__main__":
    module = ErnieGen()
    result = module.finetune(train_path='test_data/train.txt',
                             dev_path='test_data/dev.txt',
                             max_steps=30,
                             batch_size=2,
                             log_interval=10,
                             save_interval=20)
    module.export(params_path=result['last_save_path'], module_name="ernie_gen_test", author="test")
