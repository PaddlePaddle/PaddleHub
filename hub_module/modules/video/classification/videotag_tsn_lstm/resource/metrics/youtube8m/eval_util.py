# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides functions to help with evaluating models."""
import datetime
import numpy

from . import mean_average_precision_calculator as map_calculator
from . import average_precision_calculator as ap_calculator


def flatten(l):
    """ Merges a list of lists into a single list. """
    return [item for sublist in l for item in sublist]


def calculate_hit_at_one(predictions, actuals):
    """Performs a local (numpy) calculation of the hit at one.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average hit at one across the entire batch.
  """
    top_prediction = numpy.argmax(predictions, 1)
    hits = actuals[numpy.arange(actuals.shape[0]), top_prediction]
    return numpy.average(hits)


def calculate_precision_at_equal_recall_rate(predictions, actuals):
    """Performs a local (numpy) calculation of the PERR.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average precision at equal recall rate across the entire batch.
  """
    aggregated_precision = 0.0
    num_videos = actuals.shape[0]
    for row in numpy.arange(num_videos):
        num_labels = int(numpy.sum(actuals[row]))
        top_indices = numpy.argpartition(predictions[row],
                                         -num_labels)[-num_labels:]
        item_precision = 0.0
        for label_index in top_indices:
            if predictions[row][label_index] > 0:
                item_precision += actuals[row][label_index]
        item_precision /= top_indices.size
        aggregated_precision += item_precision
    aggregated_precision /= num_videos
    return aggregated_precision


def calculate_gap(predictions, actuals, top_k=20):
    """Performs a local (numpy) calculation of the global average precision.

  Only the top_k predictions are taken for each of the videos.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.
    top_k: How many predictions to use per video.

  Returns:
    float: The global average precision.
  """
    gap_calculator = ap_calculator.AveragePrecisionCalculator()
    sparse_predictions, sparse_labels, num_positives = top_k_by_class(
        predictions, actuals, top_k)
    gap_calculator.accumulate(
        flatten(sparse_predictions), flatten(sparse_labels), sum(num_positives))
    return gap_calculator.peek_ap_at_n()


def top_k_by_class(predictions, labels, k=20):
    """Extracts the top k predictions for each video, sorted by class.

  Args:
    predictions: A numpy matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    k: the top k non-zero entries to preserve in each prediction.

  Returns:
    A tuple (predictions,labels, true_positives). 'predictions' and 'labels'
    are lists of lists of floats. 'true_positives' is a list of scalars. The
    length of the lists are equal to the number of classes. The entries in the
    predictions variable are probability predictions, and
    the corresponding entries in the labels variable are the ground truth for
    those predictions. The entries in 'true_positives' are the number of true
    positives for each class in the ground truth.

  Raises:
    ValueError: An error occurred when the k is not a positive integer.
  """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    k = min(k, predictions.shape[1])
    num_classes = predictions.shape[1]
    prediction_triplets = []
    for video_index in range(predictions.shape[0]):
        prediction_triplets.extend(
            top_k_triplets(predictions[video_index], labels[video_index], k))
    out_predictions = [[] for v in range(num_classes)]
    out_labels = [[] for v in range(num_classes)]
    for triplet in prediction_triplets:
        out_predictions[triplet[0]].append(triplet[1])
        out_labels[triplet[0]].append(triplet[2])
    out_true_positives = [numpy.sum(labels[:, i]) for i in range(num_classes)]

    return out_predictions, out_labels, out_true_positives


def top_k_triplets(predictions, labels, k=20):
    """Get the top_k for a 1-d numpy array. Returns a sparse list of tuples in
  (prediction, class) format"""
    m = len(predictions)
    k = min(k, m)
    indices = numpy.argpartition(predictions, -k)[-k:]
    return [(index, predictions[index], labels[index]) for index in indices]


class EvaluationMetrics(object):
    """A class to store the evaluation metrics."""

    def __init__(self, num_class, top_k):
        """Construct an EvaluationMetrics object to store the evaluation metrics.

    Args:
      num_class: A positive integer specifying the number of classes.
      top_k: A positive integer specifying how many predictions are considered per video.

    Raises:
      ValueError: An error occurred when MeanAveragePrecisionCalculator cannot
        not be constructed.
    """
        self.sum_hit_at_one = 0.0
        self.sum_perr = 0.0
        self.sum_loss = 0.0
        self.map_calculator = map_calculator.MeanAveragePrecisionCalculator(
            num_class)
        self.global_ap_calculator = ap_calculator.AveragePrecisionCalculator()
        self.top_k = top_k
        self.num_examples = 0

    #def accumulate(self, predictions, labels, loss):
    def accumulate(self, loss, predictions, labels):
        """Accumulate the metrics calculated locally for this mini-batch.

    Args:
      predictions: A numpy matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      labels: A numpy matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.
      loss: A numpy array containing the loss for each sample.

    Returns:
      dictionary: A dictionary storing the metrics for the mini-batch.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
        does not match.
    """
        batch_size = labels.shape[0]
        mean_hit_at_one = calculate_hit_at_one(predictions, labels)
        mean_perr = calculate_precision_at_equal_recall_rate(
            predictions, labels)
        mean_loss = numpy.mean(loss)

        # Take the top 20 predictions.
        sparse_predictions, sparse_labels, num_positives = top_k_by_class(
            predictions, labels, self.top_k)
        self.map_calculator.accumulate(sparse_predictions, sparse_labels,
                                       num_positives)
        self.global_ap_calculator.accumulate(
            flatten(sparse_predictions), flatten(sparse_labels),
            sum(num_positives))

        self.num_examples += batch_size
        self.sum_hit_at_one += mean_hit_at_one * batch_size
        self.sum_perr += mean_perr * batch_size
        self.sum_loss += mean_loss * batch_size

        return {
            "hit_at_one": mean_hit_at_one,
            "perr": mean_perr,
            "loss": mean_loss
        }

    def get(self):
        """Calculate the evaluation metrics for the whole epoch.

    Raises:
      ValueError: If no examples were accumulated.

    Returns:
      dictionary: a dictionary storing the evaluation metrics for the epoch. The
        dictionary has the fields: avg_hit_at_one, avg_perr, avg_loss, and
        aps (default nan).
    """
        if self.num_examples <= 0:
            raise ValueError("total_sample must be positive.")
        avg_hit_at_one = self.sum_hit_at_one / self.num_examples
        avg_perr = self.sum_perr / self.num_examples
        avg_loss = self.sum_loss / self.num_examples

        aps = self.map_calculator.peek_map_at_n()
        gap = self.global_ap_calculator.peek_ap_at_n()

        epoch_info_dict = {}
        return {
            "avg_hit_at_one": avg_hit_at_one,
            "avg_perr": avg_perr,
            "avg_loss": avg_loss,
            "aps": aps,
            "gap": gap
        }

    def clear(self):
        """Clear the evaluation metrics and reset the EvaluationMetrics object."""
        self.sum_hit_at_one = 0.0
        self.sum_perr = 0.0
        self.sum_loss = 0.0
        self.map_calculator.clear()
        self.global_ap_calculator.clear()
        self.num_examples = 0
