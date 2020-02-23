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

#include <iostream>
#include <string>

#include "mask_detector.h" // NOLINT

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 4) {
    std::cout << "Usage:"
              << "./mask_detector ./models/ ./images/test.png"
              << std::endl;
    return -1;
  }

  bool use_gpu = (argc == 4 ? std::stoi(argv[3]) : false);
  auto det_model_dir = std::string(argv[1]) + "/pyramidbox_lite";
  auto cls_model_dir = std::string(argv[1]) + "/mask_detector";
  auto image_path = argv[2];

  // Init Detection Model
  float det_shrink = 0.6;
  float det_threshold = 0.7;
  std::vector<float> det_means = {104, 177, 123};
  std::vector<float> det_scale = {0.007843, 0.007843, 0.007843};
  FaceDetector detector(
      det_model_dir,
      det_means,
      det_scale,
      use_gpu,
      det_threshold);

  // Init Classification Model
  std::vector<float> cls_means = {0.5, 0.5, 0.5};
  std::vector<float> cls_scale = {1.0, 1.0, 1.0};
  MaskClassifier classifier(
      cls_model_dir,
      cls_means,
      cls_scale,
      use_gpu);

  // Load image
  cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
  // Prediction result
  std::vector<FaceResult> results;
  // Stage1: Face detection
  detector.Predict(img, &results, det_shrink);
  // Stage2: Mask wearing classification
  classifier.Predict(&results);

  for (const FaceResult& item : results) {
    printf("{left=%d, right=%d, top=%d, bottom=%d},"
           " class_id=%d, confidence=%.5f\n",
           item.rect[0],
           item.rect[1],
           item.rect[2],
           item.rect[3],
           item.class_id,
           item.score);
  }

  // Visualization result
  cv::Mat vis_img;
  VisualizeResult(img, results, &vis_img);
  cv::imwrite("result.jpg", vis_img);

  return 0;
}
