# Transfer Learning
Transfer Learning是xxxx
更多关于Transfer Learning的知识，请参考
## CV教程
以猫狗分类为例子，我们可以快速的使用一个通过ImageNet训练过的ResNet进行finetune
```python
import paddle.fluid as fluid
import paddle_hub as hub

resnet = hub.Module(key = "resnet_v2_50_imagenet")
input_dict, output_dict, program = resnet.context(sign_name = "feature_map")
img_mode, img_size, img_order = resnet.data_config()
reader = hub.ImageClassifierReader(mode = img_mode, shape = img_shape, order = img_order, dataset = hub.dataset.flowers(), batch_size = 32)
with fluid.program_guard(program):
	img = input_dict["image"]
	feature_map = output_dict["feature_map"]
	label = fluid.layers.data(name = "label", shape = [1], dtype = "int64")
	task = hub.DNNClassifier(input = feature_map, hidden_units = [10], acts = ["softmax"])

finetune_config = {"epochs" : 100}
hub.finetune_and_eval(task = task, reader = reader.train(), config = finetune_config)
```
## NLP教程
```python
import paddle.fluid as fluid
import paddle_hub as hub

resnet = hub.Module(key = "resnet_v2_50_imagenet")
input_dict, output_dict, program = resnet.context(sign_name = "feature_map")
img_mode, img_size, img_order = resnet.data_config()
reader = hub.ImageClassifierReader(mode = img_mode, shape = img_shape, order = img_order, dataset = hub.dataset.flowers(), batch_size = 32)
with fluid.program_guard(program):
	img = input_dict["image"]
	feature_map = output_dict["feature_map"]
	label = fluid.layers.data(name = "label", shape = [1], dtype = "int64")
	task = hub.DNNClassifier(input = feature_map, hidden_units = [10], acts = ["softmax"])

finetune_config = {"epochs" : 100}
hub.finetune_and_eval(task = task, reader = reader.train(), config = finetune_config)
```
