==================
迁移学习
==================

迁移学习 (Transfer Learning) 是属于深度学习的一个子研究领域，该研究领域的目标在于利用数据、任务、或模型之间的相似性，将在旧领域学习过的知识，迁移应用于新领域中。通俗的来讲，迁移学习就是运用已有的知识来学习新的知识，例如学会了骑自行车的人也能较快的学会骑电动车。较为常用的一种迁移学习方式是利用预训练模型进行微调，即用户基于当前任务的场景从PaddleHub中选择已训练成功的模型进行新任务训练，且该模型曾经使用的数据集与新场景的数据集情况相近，此时仅需要在当前任务场景的训练过程中使用新场景的数据对模型参数进行微调（**Fine-tune**），即可完成训练任务。迁移学习吸引了很多研究者投身其中，因为它能够很好的解决深度学习中的以下几个问题：  

* 一些研究领域只有少量标注数据，且数据标注成本较高，不足以训练一个足够鲁棒的神经网络。
* 大规模神经网络的训练依赖于大量的计算资源，这对于一般用户而言难以实现。
* 应对于普适化需求的模型，在特定应用上表现不尽如人意。  

为了让开发者更便捷地应用迁移学习，飞桨开源了预训练模型管理工具 PaddleHub。开发者仅仅使用十余行的代码，就能完成迁移学习。本文将为读者全面介绍使用PaddleHub完成迁移学习的方法。

.. image:: ../imgs/paddlehub_finetune.gif 
   :width: 900px
   :align: center

.. toctree::
   :maxdepth: 2

   finetune/sequence_labeling.md
   finetune/text_matching.md
   finetune/image_classification.md
   finetune/image_colorization.md
   finetune/style_transfer.md
   finetune/semantic_segmentation.md
   finetune/audio_classification.md
   finetune/customized_dataset.md