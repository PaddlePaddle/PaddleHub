import os
import numpy as np
import processor
import paddle_hub as hub
import paddle
import paddle.fluid as fluid
from mobilenet_ssd import mobile_net


def build_program():
    image_shape = [3, 300, 300]
    class_num = 21
    image = fluid.layers.data(dtype="float32", shape=image_shape, name="image")
    gt_box = fluid.layers.data(
        dtype="float32", shape=[4], name="gtbox", lod_level=1)
    gt_label = fluid.layers.data(
        dtype="int32", shape=[1], name="label", lod_level=1)
    difficult = fluid.layers.data(
        dtype="int32", shape=[1], name="difficult", lod_level=1)
    with fluid.unique_name.guard():
        locs, confs, box, box_var = mobile_net(class_num, image, image_shape)
        nmsed_out = fluid.layers.detection_output(
            locs, confs, box, box_var, nms_threshold=0.45)

    return image, nmsed_out


def create_module():
    image, nmsed_out = build_program()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    pretrained_model = "resources/ssd_mobilenet_v1_pascalvoc"

    def if_exist(var):
        return os.path.exists(os.path.join(pretrained_model, var.name))

    fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    assets = ["resources/label_list.txt"]
    sign = hub.create_signature(
        "object_detection", inputs=[image], outputs=[nmsed_out])
    hub.create_module(
        sign_arr=[sign],
        module_dir="hub_module_ssd",
        exe=exe,
        processor=processor.Processor,
        assets=assets)


if __name__ == '__main__':
    create_module()
