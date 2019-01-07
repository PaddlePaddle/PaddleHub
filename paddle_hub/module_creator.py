#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import paddle_hub.module_desc_pb2 as modulepb
import paddle.fluid as fluid
from paddle_hub.utils import to_list
from paddle_hub.signature import Signature
from paddle_hub.module import mkdir

import os


def create_module(sign_arr, program, path=None, assets=None):
    assert isinstance(
        program, fluid.Program), "program should be instance of fluid.Program"
    assert sign_arr, "signarture array should not be None"

    if not path:
        path = os.path.join(".", "hub_module")
    # create module path for saving
    mkdir(path)

    module = modulepb.ModuleDesc()
    program = program.clone()

    # TODO(wuzewu): save assets data
    if not assets:
        module.contain_assets = False
    else:
        module.contain_assets = True
        os.makedirs(os.path.join(path, "assets"))

    # save fluid Parameter
    for param in program.global_block().iter_parameters():
        parameter = module.parameters.add()
        parameter.name = param.name
        parameter.learning_rate = param.optimize_attr["learning_rate"]
        parameter.trainable = param.trainable

    # save signarture info
    sign_map = module.sign2var
    sign_arr = to_list(sign_arr)
    for sign in sign_arr:
        assert isinstance(sign,
                          Signature), "sign_arr should be list of Signature"

        if sign.get_name() in sign_map:
            raise "Error! sign_arr contains repeat signatrue %s" % sign

        var = sign_map[sign.get_name()]
        feed_desc = var.feed_desc
        fetch_desc = var.fetch_desc
        for input in sign.get_inputs():
            feed_var = feed_desc.add()
            feed_var.var_name = input.name

        for output in sign.get_outputs():
            fetch_var = fetch_desc.add()
            fetch_var.var_name = output.name

    # save inference program
    exe = fluid.Executor(place=fluid.CPUPlace())
    model_path = os.path.join(path, "model")
    os.makedirs(model_path)
    first_sign = sign_arr[0]
    fluid.io.save_inference_model(
        model_path,
        feeded_var_names=[var.name for var in first_sign.get_inputs()],
        target_vars=first_sign.get_outputs(),
        executor=exe)

    # save to dist
    data = module.SerializeToString()
    metafile = os.path.join(path, "module_desc.pb")
    with open(metafile, "wb") as f:
        f.write(data)
