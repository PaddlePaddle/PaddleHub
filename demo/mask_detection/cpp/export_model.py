#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import paddlehub as hub

# Load mask detector module from PaddleHub
module = hub.Module(name="pyramidbox_lite_server_mask", version='1.1.0')
# Export inference model for deployment
module.processor.save_inference_model("./pyramidbox_lite_server_mask")
# rename of params
classify_param = "./pyramidbox_lite_server_mask/pyramidbox_lite/__param__"
detection_param = "./pyramidbox_lite_server_mask/mask_detector/__param__"
if os.path.isfile(detection_param):
    os.system("mv " + detection_param +
              " ./pyramidbox_lite_server_mask/mask_detector/__params__")
if os.path.isfile(classify_param):
    os.system("mv " + classify_param +
              " ./pyramidbox_lite_server_mask/pyramidbox_lite/__params__")
print("pyramidbox_lite_server_mask module export done!")

# Load mask detector (mobile version) module from PaddleHub
module = hub.Module(name="pyramidbox_lite_mobile_mask", version="1.1.0")
# Export inference model for deployment
module.processor.save_inference_model("./pyramidbox_lite_mobile_mask")

# rename of params
classify_param = "./pyramidbox_lite_mobile_mask/pyramidbox_lite/__param__"
detection_param = "./pyramidbox_lite_mobile_mask/mask_detector/__param__"
if os.path.isfile(detection_param):
    os.system("mv " + detection_param +
              " ./pyramidbox_lite_mobile_mask/mask_detector/__params__")
if os.path.isfile(classify_param):
    os.system("mv " + classify_param +
              " ./pyramidbox_lite_mobile_mask/pyramidbox_lite/__params__")
print("pyramidbox_lite_mobile_mask module export done!")
