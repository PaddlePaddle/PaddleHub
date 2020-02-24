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

# include "mask_detector.h"

// Normalize the image by (pix - mean) * scale
void NormalizeImage(
    const std::vector<float> &mean,
    const std::vector<float> &scale,
    cv::Mat& im, // NOLINT
    float* input_buffer) {
  int height = im.rows;
  int width = im.cols;
  int stride = width * height;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int base = h * width + w;
      input_buffer[base + 0 * stride] =
          (im.at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
      input_buffer[base + 1 * stride] =
          (im.at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
      input_buffer[base + 2 * stride] =
          (im.at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}

// Load Model and return model predictor
void LoadModel(
    const std::string& model_dir,
    bool use_gpu,
    std::unique_ptr<paddle::PaddlePredictor>* predictor) {
  // Config the model info
  paddle::AnalysisConfig config;
  config.SetModel(model_dir + "/__model__",
                  model_dir + "/__params__");
  if (use_gpu) {
      config.EnableUseGpu(100, 0);
  } else {
      config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // Memory optimization
  config.EnableMemoryOptim();
  *predictor = std::move(CreatePaddlePredictor(config));
}


// Visualiztion MaskDetector results
void VisualizeResult(const cv::Mat& img,
                     const std::vector<FaceResult>& results,
                     cv::Mat* vis_img) {
  for (int i = 0; i < results.size(); ++i) {
    int w = results[i].rect[1] - results[i].rect[0];
    int h = results[i].rect[3] - results[i].rect[2];
    cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[2], w, h);

    // Configure color and text size
    cv::Scalar roi_color;
    std::string text;
    if (results[i].class_id == 1) {
      text = "MASK:  ";
      roi_color = cv::Scalar(0, 255, 0);
    } else {
      text = "NO MASK:  ";
      roi_color = cv::Scalar(0, 0, 255);
    }
    text += std::to_string(static_cast<int>(results[i].confidence * 100)) + "%";
    int font_face = cv::FONT_HERSHEY_TRIPLEX;
    double font_scale = 1.f;
    float thickness = 1;
    cv::Size text_size = cv::getTextSize(text,
                                         font_face,
                                         font_scale,
                                         thickness,
                                         nullptr);
    float new_font_scale = roi.width * font_scale / text_size.width;
    text_size = cv::getTextSize(text,
                               font_face,
                               new_font_scale,
                               thickness,
                               nullptr);
    cv::Point origin;
    origin.x = roi.x;
    origin.y = roi.y;

    // Configure text background
    cv::Rect text_back = cv::Rect(results[i].rect[0],
    results[i].rect[2] - text_size.height,
    text_size.width,
    text_size.height);

    // Draw roi object, text, and background
    *vis_img = img;
    cv::rectangle(*vis_img, roi, roi_color, 2);
    cv::rectangle(*vis_img, text_back, cv::Scalar(225, 225, 225), -1);
    cv::putText(*vis_img,
                text,
                origin,
                font_face,
                new_font_scale,
                cv::Scalar(0, 0, 0),
                thickness);
  }
}



void FaceDetector::Preprocess(const cv::Mat& image_mat, float shrink) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = image_mat.clone();
  cv::resize(im, im, cv::Size(), shrink, shrink, cv::INTER_CUBIC);
  im.convertTo(im, CV_32FC3, 1.0);
  int rc = im.channels();
  int rh = im.rows;
  int rw = im.cols;
  input_shape_ = {1, rc, rh, rw};
  input_data_.resize(1 * rc * rh * rw);
  float* buffer = input_data_.data();
  NormalizeImage(mean_, scale_, im, input_data_.data());
}

void FaceDetector::Postprocess(
    const cv::Mat& raw_mat,
    float shrink,
    std::vector<FaceResult>* result) {
  result->clear();
  int rect_num = 0;
  int rh = input_shape_[2];
  int rw = input_shape_[3];
  int total_size = output_data_.size() / 6;
  for (int j = 0; j < total_size; ++j) {
    // Class id
    int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
    // Confidence score
    float score = output_data_[1 + j * 6];
    int xmin = (output_data_[2 + j * 6] * rw) / shrink;
    int ymin = (output_data_[3 + j * 6] * rh) / shrink;
    int xmax = (output_data_[4 + j * 6] * rw) / shrink;
    int ymax = (output_data_[5 + j * 6] * rh) / shrink;
    int wd = xmax - xmin;
    int hd = ymax - ymin;
    if (score > threshold_) {
      auto roi = cv::Rect(xmin, ymin, wd, hd) &
                  cv::Rect(0, 0, rw / shrink, rh / shrink);
      // A view ref to original mat
      cv::Mat roi_ref(raw_mat, roi);
      FaceResult result_item;
      result_item.rect = {xmin, xmax, ymin, ymax};
      result_item.roi_rect = roi_ref;
      result->push_back(result_item);
    }
  }
}

void FaceDetector::Predict(const cv::Mat& im,
                                  std::vector<FaceResult>* result,
                                  float shrink) {
  // Preprocess image
  Preprocess(im, shrink);
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  auto in_tensor = predictor_->GetInputTensor(input_names[0]);
  in_tensor->Reshape(input_shape_);
  in_tensor->copy_from_cpu(input_data_.data());
  // Run predictor
  predictor_->ZeroCopyRun();
  // Get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto out_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = out_tensor->shape();
  // Calculate output length
  int output_size = 1;
  for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
  }
  output_data_.resize(output_size);
  out_tensor->copy_to_cpu(output_data_.data());
  // Postprocessing result
  Postprocess(im, shrink, result);
}

inline void MaskClassifier::Preprocess(std::vector<FaceResult>* faces) {
  int batch_size = faces->size();
  input_shape_ = {
      batch_size,
      EVAL_CROP_SIZE_[0],
      EVAL_CROP_SIZE_[1],
      EVAL_CROP_SIZE_[2]
  };
  // Reallocate input buffer
  int input_size = 1;
  for (int x : input_shape_) {
    input_size *= x;
  }
  input_data_.resize(input_size);
  auto buffer_base = input_data_.data();
  for (int i = 0; i < batch_size; ++i) {
    cv::Mat im = (*faces)[i].roi_rect;
    // Resize
    int rc = im.channels();
    int rw = im.cols;
    int rh = im.rows;
    cv::Size resize_size(input_shape_[3], input_shape_[2]);
    if (rw != input_shape_[3] || rh != input_shape_[2]) {
      cv::resize(im, im, resize_size, 0.f, 0.f, cv::INTER_CUBIC);
    }
    im.convertTo(im, CV_32FC3, 1.0 / 256.0);
    rc = im.channels();
    rw = im.cols;
    rh = im.rows;
    float* buffer_i = buffer_base + i * rc * rw * rh;
    NormalizeImage(mean_, scale_, im, buffer_i);
  }
}

void MaskClassifier::Postprocess(std::vector<FaceResult>* faces) {
  float* data = output_data_.data();
  int batch_size = faces->size();
  int out_num = output_data_.size();
  for (int i = 0; i < batch_size; ++i) {
    auto out_addr = data + (out_num / batch_size) * i;
    int best_class_id = 0;
    float best_class_score = *(best_class_id + out_addr);
    for (int j = 0; j < (out_num / batch_size); ++j) {
      auto infer_class = j;
      auto score = *(j + out_addr);
      if (score > best_class_score) {
        best_class_id = infer_class;
        best_class_score = score;
      }
    }
    (*faces)[i].class_id = best_class_id;
    (*faces)[i].confidence = best_class_score;
  }
}

void MaskClassifier::Predict(std::vector<FaceResult>* faces) {
  Preprocess(faces);
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  auto in_tensor = predictor_->GetInputTensor(input_names[0]);
  in_tensor->Reshape(input_shape_);
  in_tensor->copy_from_cpu(input_data_.data());
  // Run predictor
  predictor_->ZeroCopyRun();
  // Get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto out_tensor = predictor_->GetOutputTensor(output_names[1]);
  std::vector<int> output_shape = out_tensor->shape();
  // Calculate output length
  int output_size = 1;
  for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
  }
  output_data_.resize(output_size);
  out_tensor->copy_to_cpu(output_data_.data());
  Postprocess(faces);
}
