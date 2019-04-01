from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import functools
import argparse
import paddle
import paddle.fluid as fluid
import nets
import paddle_hub as hub
import processor
from utility import add_arguments, print_arguments
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model',            str,   "ResNet50",          "Set the network to use.")
add_arg('pretrained_model', str,   None,                "Whether to use pretrained model.")
# yapf: enable


def build_program(args):
    image_shape = [3, 224, 224]
    model_name = args.model
    model = nets.__dict__[model_name]()
    image = fluid.layers.data(name="image", shape=image_shape, dtype="float32")
    predition, feature_map = model.net(input=image, class_dim=1000)

    return image, predition, feature_map


def create_module(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model

    image, predition, feature_map = build_program(args)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load pretrained model param
    def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))

    fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    # create paddle hub module
    assets = ["resources/label_list.txt"]
    sign1 = hub.create_signature(
        "classification", inputs=[image], outputs=[predition], for_predict=True)
    sign2 = hub.create_signature(
        "feature_map", inputs=[image], outputs=[feature_map])
    hub.create_module(
        sign_arr=[sign1, sign2],
        module_dir="hub_module_" + args.model,
        module_info="resources/module_info.yml",
        processor=processor.Processor,
        assets=assets)


def main():
    args = parser.parse_args()
    assert args.model in nets.__all__, "model is not in list %s" % nets.__all__
    print_arguments(args)
    create_module(args)


if __name__ == '__main__':
    main()
