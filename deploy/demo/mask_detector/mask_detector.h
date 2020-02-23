//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "paddle_inference_api.h" // NOLINT

// MaskDetector Result
struct FaceResult {
  // Detection result: face rectangle
  std::vector<int> rect;
  // Detection result: cv::Mat of face rectange
  cv::Mat roi_rect;
  // Classification result: confidence
  float score;
  // Classification result : class id
  int class_id;
};

// Load Paddle Inference Model
void LoadModel(
    const std::string& model_dir,
    bool use_gpu,
    std::unique_ptr<paddle::PaddlePredictor>* predictor);

// Visualiztion MaskDetector results
void VisualizeResult(const cv::Mat& img,
                     const std::vector<FaceResult>& results,
                     cv::Mat* vis_img);

class FaceDetector {
 public:
  explicit FaceDetector(const std::string& model_dir,
                        const std::vector<float>& mean,
                        const std::vector<float>& scale,
                        bool use_gpu = false,
                        float threshold = 0.7) :
      mean_(mean),
      scale_(scale),
      threshold_(threshold) {
    LoadModel(model_dir, use_gpu, &predictor_);
  }

  // Run predictor
  void Predict(
      const cv::Mat& img,
      std::vector<FaceResult>* result,
      float shrink);

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat, float shrink);
  // Postprocess result
  void Postprocess(
      const cv::Mat& raw_mat,
      float shrink,
      std::vector<FaceResult>* result);

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<float> input_data_;
  std::vector<float> output_data_;
  std::vector<int> input_shape_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  float threshold_;
};

class MaskClassifier {
 public:
  explicit MaskClassifier(const std::string& model_dir,
                      const std::vector<float>& mean,
                      const std::vector<float>& scale,
                      bool use_gpu = false) :
  mean_(mean),
  scale_(scale) {
    LoadModel(model_dir, use_gpu, &predictor_);
  }

  void Predict(std::vector<FaceResult>* faces);

 private:
  void Preprocess(std::vector<FaceResult>* faces);

  void Postprocess(std::vector<FaceResult>* faces);

  std::unique_ptr<paddle::PaddlePredictor> predictor_;
  std::vector<float> input_data_;
  std::vector<int> input_shape_;
  std::vector<float> output_data_;
  const std::vector<int> EVAL_CROP_SIZE_ = {3, 128, 128};
  std::vector<float> mean_;
  std::vector<float> scale_;
};
