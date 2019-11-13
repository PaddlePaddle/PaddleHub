#coding:utf-8
#  Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import collections
import math
import six
import json

from collections import OrderedDict

import numpy as np
import paddle.fluid as fluid
from .basic_task import BasicTask
from paddlehub.common.logger import logger
from paddlehub.reader import tokenization
from paddlehub.finetune.evaluator import squad1_evaluate
from paddlehub.finetune.evaluator import squad2_evaluate
from paddlehub.finetune.evaluator import cmrc2018_evaluate


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_final_text(pred_text, orig_text, do_lower_case, is_english):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    if is_english:
        tok_text = " ".join(tokenizer.tokenize(orig_text))
    else:
        tok_text = "".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # using in debug
        # logger.info(
        #     "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        # using in debug
        # logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
        #                 orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        # using in debug
        # logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        # using in debug
        # tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      version_2_with_negative, null_score_diff_threshold,
                      is_english):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple("PrelimPrediction", [
        "feature_index", "start_index", "end_index", "start_logit", "end_logit"
    ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in unique_id_to_result:
                logger.info(
                    "As using pyreader, the last one batch is so small that the feature %s in the last batch is discarded "
                    % feature.unique_id)
                continue
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[
                    0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        if not prelim_predictions:
            logger.warning(("not prelim_predictions:", example.qas_id))
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(
                    orig_doc_end + 1)]
                if is_english:
                    tok_text = " ".join(tok_tokens)
                else:
                    tok_text = "".join(tok_tokens)
                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                if is_english:
                    orig_text = " ".join(orig_tokens)
                else:
                    orig_text = "".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case,
                                            is_english)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null
            if best_non_null_entry:
                score_diff -= best_non_null_entry.start_logit + best_non_null_entry.end_logit
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json
    """Write final predictions to the json file and log-odds of null if needed."""
    with open(output_prediction_file, "w") as writer:
        logger.info("Writing predictions to: %s" % (output_prediction_file))
        writer.write(
            json.dumps(all_predictions, indent=4, ensure_ascii=is_english) +
            "\n")

    with open(output_nbest_file, "w") as writer:
        logger.info("Writing nbest to: %s" % (output_nbest_file))
        writer.write(
            json.dumps(all_nbest_json, indent=4, ensure_ascii=is_english) +
            "\n")

    if version_2_with_negative:
        logger.info("Writing null_log_odds to: %s" % (output_nbest_file))
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(
                json.dumps(scores_diff_json, indent=4, ensure_ascii=is_english)
                + "\n")


class ReadingComprehensionTask(BasicTask):
    def __init__(self,
                 feature,
                 feed_list,
                 data_reader,
                 startup_program=None,
                 config=None,
                 metrics_choices=None,
                 sub_task="squad",
                 null_score_diff_threshold=0.0,
                 n_best_size=20,
                 max_answer_length=30):

        main_program = feature.block.program
        super(ReadingComprehensionTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            metrics_choices=metrics_choices)
        self.feature = feature
        self.data_reader = data_reader
        self.sub_task = sub_task.lower()
        self.version_2_with_negative = (self.sub_task == "squad2.0")
        if self.sub_task in ["squad2.0", "squad"]:
            self.is_english = True
        elif self.sub_task in ["cmrc2018", "drcd"]:
            self.is_english = False
        else:
            raise Exception("No language type of data set is sepecified")

        self.null_score_diff_threshold = null_score_diff_threshold
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length

    def _build_net(self):
        self.unique_ids = fluid.layers.data(
            name="unique_ids", shape=[-1, 1], lod_level=0, dtype="int64")

        logits = fluid.layers.fc(
            input=self.feature,
            size=2,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="cls_seq_label_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_seq_label_out_b",
                initializer=fluid.initializer.Constant(0.)))

        logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
        start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

        batch_ones = fluid.layers.fill_constant_batch_size_like(
            input=start_logits, dtype='int64', shape=[1], value=1)
        num_seqs = fluid.layers.reduce_sum(input=batch_ones)

        return [start_logits, end_logits, num_seqs]

    def _add_label(self):
        start_positions = fluid.layers.data(
            name="start_positions", shape=[-1, 1], lod_level=0, dtype="int64")
        end_positions = fluid.layers.data(
            name="end_positions", shape=[-1, 1], lod_level=0, dtype="int64")
        return [start_positions, end_positions]

    def _add_loss(self):
        start_positions = self.labels[0]
        end_positions = self.labels[1]

        start_logits = self.outputs[0]
        end_logits = self.outputs[1]

        start_loss = fluid.layers.softmax_with_cross_entropy(
            logits=start_logits, label=start_positions)
        start_loss = fluid.layers.mean(x=start_loss)
        end_loss = fluid.layers.softmax_with_cross_entropy(
            logits=end_logits, label=end_positions)
        end_loss = fluid.layers.mean(x=end_loss)
        total_loss = (start_loss + end_loss) / 2.0
        return total_loss

    def _add_metrics(self):
        return []

    @property
    def feed_list(self):
        feed_list = [varname for varname in self._base_feed_list
                     ] + [self.unique_ids.name]
        if self.is_train_phase or self.is_test_phase:
            feed_list += [label.name for label in self.labels]
        return feed_list

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [
                self.loss.name, self.outputs[-1].name, self.unique_ids.name,
                self.outputs[0].name, self.outputs[1].name
            ]
        elif self.is_predict_phase:
            return [
                self.unique_ids.name,
            ] + [output.name for output in self.outputs]

    def _calculate_metrics(self, run_states):
        total_cost, total_num_seqs, all_results = [], [], []
        run_step = 0
        RawResult = collections.namedtuple(
            "RawResult", ["unique_id", "start_logits", "end_logits"])
        for run_state in run_states:
            np_loss = run_state.run_results[0]
            np_num_seqs = run_state.run_results[1]
            total_cost.extend(np_loss * np_num_seqs)
            total_num_seqs.extend(np_num_seqs)
            run_step += run_state.run_step
            if self.is_test_phase:
                np_unique_ids = run_state.run_results[2]
                np_start_logits = run_state.run_results[3]
                np_end_logits = run_state.run_results[4]
                for idx in range(np_unique_ids.shape[0]):
                    unique_id = int(np_unique_ids[idx])
                    start_logits = [float(x) for x in np_start_logits[idx].flat]
                    end_logits = [float(x) for x in np_end_logits[idx].flat]
                    all_results.append(
                        RawResult(
                            unique_id=unique_id,
                            start_logits=start_logits,
                            end_logits=end_logits))

        run_time_used = time.time() - run_states[0].run_time_begin
        run_speed = run_step / run_time_used
        avg_loss = np.sum(total_cost) / np.sum(total_num_seqs)
        scores = OrderedDict()
        # If none of metrics has been implemented, loss will be used to evaluate.
        if self.is_test_phase:
            output_prediction_file = os.path.join(self.config.checkpoint_dir,
                                                  "predictions.json")
            output_nbest_file = os.path.join(self.config.checkpoint_dir,
                                             "nbest_predictions.json")
            output_null_log_odds_file = os.path.join(self.config.checkpoint_dir,
                                                     "null_odds.json")
            all_examples = self.data_reader.all_examples[self.phase]
            all_features = self.data_reader.all_features[self.phase]
            write_predictions(
                all_examples=all_examples,
                all_features=all_features,
                all_results=all_results,
                n_best_size=self.n_best_size,
                max_answer_length=self.max_answer_length,
                do_lower_case=True,
                output_prediction_file=output_prediction_file,
                output_nbest_file=output_nbest_file,
                output_null_log_odds_file=output_null_log_odds_file,
                version_2_with_negative=self.version_2_with_negative,
                null_score_diff_threshold=self.null_score_diff_threshold,
                is_english=self.is_english)
            if self.phase == 'val' or self.phase == 'dev':
                with open(
                        self.data_reader.dataset.dev_file, 'r',
                        encoding="utf8") as dataset_file:
                    dataset_json = json.load(dataset_file)
                    dataset = dataset_json['data']
            elif self.phase == 'test':
                with open(
                        self.data_reader.dataset.test_file, 'r',
                        encoding="utf8") as dataset_file:
                    dataset_json = json.load(dataset_file)
                    dataset = dataset_json['data']
            else:
                raise Exception("Error phase: %s when runing _calculate_metrics"
                                % self.phase)
            with open(
                    output_prediction_file, 'r',
                    encoding="utf8") as prediction_file:
                predictions = json.load(prediction_file)

            if self.sub_task == "squad":
                scores = squad1_evaluate.evaluate(dataset, predictions)
            elif self.sub_task == "squad2.0":
                with open(
                        output_null_log_odds_file, 'r',
                        encoding="utf8") as odds_file:
                    na_probs = json.load(odds_file)
                scores = squad2_evaluate.evaluate(dataset, predictions,
                                                  na_probs)
            elif self.sub_task in ["cmrc2018", "drcd"]:
                scores = cmrc2018_evaluate.get_eval(dataset, predictions)
        return scores, avg_loss, run_speed

    def _predict_end_event(self, run_states):
        all_results = []
        RawResult = collections.namedtuple(
            "RawResult", ["unique_id", "start_logits", "end_logits"])
        for run_state in run_states:
            np_unique_ids = run_state.run_results[0]
            np_start_logits = run_state.run_results[1]
            np_end_logits = run_state.run_results[2]
            for idx in range(np_unique_ids.shape[0]):
                unique_id = int(np_unique_ids[idx])
                start_logits = [float(x) for x in np_start_logits[idx].flat]
                end_logits = [float(x) for x in np_end_logits[idx].flat]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits))
        # If none of metrics has been implemented, loss will be used to evaluate.
        output_prediction_file = os.path.join(self.config.checkpoint_dir,
                                              "predict_predictions.json")
        output_nbest_file = os.path.join(self.config.checkpoint_dir,
                                         "predict_nbest_predictions.json")
        output_null_log_odds_file = os.path.join(self.config.checkpoint_dir,
                                                 "predict_null_odds.json")
        all_examples = self.data_reader.all_examples[self.phase]
        all_features = self.data_reader.all_features[self.phase]
        write_predictions(
            all_examples=all_examples,
            all_features=all_features,
            all_results=all_results,
            n_best_size=self.n_best_size,
            max_answer_length=self.max_answer_length,
            do_lower_case=True,
            output_prediction_file=output_prediction_file,
            output_nbest_file=output_nbest_file,
            output_null_log_odds_file=output_null_log_odds_file,
            version_2_with_negative=self.version_2_with_negative,
            null_score_diff_threshold=self.null_score_diff_threshold,
            is_english=self.is_english)

        logger.info("PaddleHub predict finished.")

        logger.info("You can see the prediction in %s" % output_prediction_file)
