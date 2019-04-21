# PaddleHub Finetune API与迁移学习

## 简述
迁移学习(Transfer Learning)是属于机器学习的一个子研究领域，该研究领域的目标在于利用数据、任务、或模型之间的相似性，将在旧领域学习过的知识，迁移应用于新领域中

基于以下几个原因，迁移学习吸引了很多研究者投身其中：

* 一些研究领域只有少量标注数据，且数据标注成本较高，不足以训练一个足够鲁棒的神经网络
* 大规模神经网络的训练依赖于大量的计算资源，这对于一般用户而言难以实现
* 应对于普适化需求的模型，在特定应用上表现不尽如人意

目前在深度学习领域已经取得了较大的发展，本文让用户了解如何快速使用PaddleHub进行迁移学习。 更多关于Transfer Learning的知识，请参考：

http://cs231n.github.io/transfer-learning/

https://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf

http://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf

## PaddleHub中的迁移学习

PaddleHub提供了基于PaddlePaddle框架实现的Finetune API, 重点针对大规模预训练模型的Fine-tuning任务做了高阶的抽象，帮助用户使用最少的代码快速、稳定地完成预训练模型的fine-tuning。

教程会涵盖CV领域的图像分类迁移，和NLP文本分类迁移两种任务。

* [CV教程](https://github.com/PaddlePaddle/PaddleHub/tree/develop/docs/turtorial/cv_finetune_turtorial.md)
* [NLP教程]()
