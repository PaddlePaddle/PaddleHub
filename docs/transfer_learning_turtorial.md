# Transfer Learning

## 简述
Transfer Learning是属于机器学习的一个子研究领域，该研究领域的目标在于利用数据、任务、或模型之间的相似性，将在旧领域学习过的知识，迁移应用于新领域中

基于以下几个原因，迁移学习吸引了很多研究者投身其中：

* 一些研究领域只有少量标注数据，且数据标注成本较高，不足以训练一个足够鲁棒的神经网络
* 大规模神经网络的训练依赖于大量的计算资源，这对于一般用户而言难以实现
* 应对于普适化需求的模型，在特定应用上表现不尽如人意

目前在深度学习领域已经取得了较大的发展，本文让用户了解如何快速使用PaddleHub进行迁移学习。 更多关于Transfer Learning的知识，请参考：

http://cs231n.github.io/transfer-learning/

https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf

http://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf

## PaddleHub中的迁移学习
PaddleHub 提供了基于PaddlePaddle框架的高阶Finetune API, 对常见的预训练模型迁移学习任务进行了抽象，帮助用户使用最少的代码快速完成迁移学习。
教程会包含CV领域的图像分类迁移，和NLP文本分类迁移两种任务。

### CV教程
以猫狗分类为例子，我们可以快速的使用一个通过ImageNet训练过的ResNet进行finetune
```python
import paddlehub as hub
import paddle
import paddle.fluid as fluid

def train():
    resnet_module = hub.Module(name="resnet50_imagenet")
    input_dict, output_dict, program = resnet_module.context(
        sign_name="feature_map", trainable=True)
    dataset = hub.dataset.DogCat()
    data_reader = hub.ImageClassificationReader(
        image_width=224, image_height=224, dataset=dataset)
    with fluid.program_guard(program):
        label = fluid.layers.data(name="label", dtype="int64", shape=[1])
        img = input_dict["img"]
        feature_map = output_dict["feature_map"]

		# 运行配置
        config = hub.RunConfig(
            use_cuda=True,
            num_epoch=10,
            batch_size=32,
            strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

        feed_list = [img.name, label.name]

	# 构造多分类模型任务
        task = hub.create_img_classfiication_task(
            feature=feature_map, label=label, num_classes=dataset.num_labels)

        # finetune
        hub.finetune_and_eval(
            task, feed_list=feed_list, data_reader=data_reader, config=config)


if __name__ == "__main__":
    train()

```
