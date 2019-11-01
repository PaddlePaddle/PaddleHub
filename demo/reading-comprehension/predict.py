#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Finetuning on classification task """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import collections
import json
import io
import math
import numpy as np
import os
import six
import sys
import time

import paddle
import paddle.fluid as fluid
import paddlehub as hub
import evaluate_v1
import evaluate_v2

hub.common.logger.logger.setLevel("INFO")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint.")
parser.add_argument("--result_dir", type=str, default=None, help="Directory to predicted results to be written.")
parser.add_argument("--max_seq_len", type=int, default=384, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--use_pyreader", type=ast.literal_eval, default=False, help="Whether use pyreader to feed data.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")
parser.add_argument("--max_answer_length", type=int, default=30, help="Max answer length.")
parser.add_argument("--n_best_size", type=int, default=20, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--null_score_diff_threshold", type=float, default=0.0, help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--version_2_with_negative", type=ast.literal_eval, default=False, help="If true, the SQuAD examples contain some that do not have an answer. If using squad v2.0, it should be set true.")
args = parser.parse_args()
# yapf: enable.


def write_predictions(
        all_examples,
        all_features,
        all_results,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        ensure_ascii,
        n_best_size=20,
        max_answer_length=30,
        do_lower_case=True,
        version_2_with_negative=False,
        null_score_diff_threshold=0.0,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit",
            "end_logit"
        ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = get_best_indexes(result.start_logits, n_best_size)
            end_indexes = get_best_indexes(result.end_logits, n_best_size)
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
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
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

        # if we didn't inlude the empty option in the n-best, inlcude it
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
        # debug
        if best_non_null_entry is None:
            print("the best_non_null_entry is None, sth wrong")

        probs = compute_softmax(total_scores)

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
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(
            json.dumps(all_predictions, indent=4, ensure_ascii=ensure_ascii) +
            "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(
            json.dumps(all_nbest_json, indent=4, ensure_ascii=ensure_ascii) +
            "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case):
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
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
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
    tokenizer = hub.reader.tokenization.BasicTokenizer(
        do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
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
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def compute_softmax(scores):
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


if __name__ == '__main__':
    # Load Paddlehub bert_uncased_L-12_H-768_A-12 pretrained model
    module = hub.Module(name="bert_uncased_L-12_H-768_A-12")
    # module = hub.Module(module_dir=["./bert_uncased_L-12_H-768_A-12.hub_module"])
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use ReadingComprehensionReader to read dataset
    dataset = hub.dataset.SQUAD(
        version_2_with_negative=args.version_2_with_negative)

    reader = hub.reader.ReadingComprehensionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_length=args.max_seq_len,
        doc_stride=128,
        max_query_length=64)

    # Use "sequence_output" for token-level output.
    seq_output = outputs["sequence_output"]

    # Setup feed list for data feeder
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Select finetune strategy, setup config and finetune
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion,
        lr_scheduler="linear_decay")

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        log_interval=10,
        use_pyreader=args.use_pyreader,
        use_data_parallel=args.use_data_parallel,
        save_ckpt_interval=100,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        enable_memory_optim=True,
        strategy=strategy)

    # Define a reading comprehension finetune task by PaddleHub's API
    reading_comprehension_task = hub.ReadingComprehensionTask(
        data_reader=reader,
        feature=seq_output,
        feed_list=feed_list,
        config=config)

    # Data to be predicted
    data = dataset.predict_examples

    features = reader.convert_examples_to_features(
        examples=data, is_training=False)
    run_states = reading_comprehension_task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]

    RawResult = collections.namedtuple(
        "RawResult", ["unique_id", "start_logits", "end_logits"])
    all_results = []
    for batch_idx, batch_result in enumerate(results):
        np_unique_ids = batch_result[0]
        np_start_logits = batch_result[1]
        np_end_logits = batch_result[2]
        np_num_seqs = batch_result[3]

        for idx in range(np_unique_ids.shape[0]):
            unique_id = int(np_unique_ids[idx])
            start_logits = [float(x) for x in np_start_logits[idx].flat]
            end_logits = [float(x) for x in np_end_logits[idx].flat]
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits))

    output_prediction_file = os.path.join(args.result_dir, "predictions.json")
    output_nbest_file = os.path.join(args.result_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.result_dir, "null_odds.json")

    write_predictions(
        data,
        features,
        all_results,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        max_answer_length=args.max_answer_length,
        n_best_size=args.n_best_size,
        version_2_with_negative=args.version_2_with_negative,
        null_score_diff_threshold=args.null_score_diff_threshold)

    with io.open(dataset.predict_file, 'r', encoding="utf8") as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']

    with io.open(
            output_prediction_file, 'r', encoding="utf8") as prediction_file:
        predictions = json.load(prediction_file)

    if not args.version_2_with_negative:
        print(json.dumps(evaluate_v1.evaluate(dataset, predictions)))
    else:
        with io.open(
                output_null_log_odds_file, 'r', encoding="utf8") as odds_file:
            na_probs = json.load(odds_file)
        # Maps qid to true/false
        qid_to_has_ans = evaluate_v2.make_qid_to_has_ans(dataset)
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
        exact_raw, f1_raw = evaluate_v2.get_raw_scores(dataset, predictions)
        exact_thresh = evaluate_v2.apply_no_ans_threshold(
            exact_raw, na_probs, qid_to_has_ans, na_prob_thresh=1.0)
        f1_thresh = evaluate_v2.apply_no_ans_threshold(
            f1_raw, na_probs, qid_to_has_ans, na_prob_thresh=1.0)
        out_eval = evaluate_v2.make_eval_dict(exact_thresh, f1_thresh)

        if has_ans_qids:
            has_ans_eval = evaluate_v2.make_eval_dict(
                exact_thresh, f1_thresh, qid_list=has_ans_qids)
            evaluate_v2.merge_eval(out_eval, has_ans_eval, 'HasAns')
        if no_ans_qids:
            no_ans_eval = evaluate_v2.make_eval_dict(
                exact_thresh, f1_thresh, qid_list=no_ans_qids)
            evaluate_v2.merge_eval(out_eval, no_ans_eval, 'NoAns')

        evaluate_v2.find_all_best_thresh(out_eval, predictions, exact_raw,
                                         f1_raw, na_probs, qid_to_has_ans)
        print(json.dumps(out_eval, indent=4))
