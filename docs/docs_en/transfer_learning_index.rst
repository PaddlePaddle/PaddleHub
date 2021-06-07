==================
Transfer Learning
==================

Transfer Learning is a subfield of deep learning that aims to use similarities in data, tasks, or models to transfer knowledge learned in old fields to new fields. In other words, transfer learning refers to the use of existing knowledge to learn new knowledge. For example, people who have learned to ride a bicycle can learn to ride an electric bike faster. A common method of transfer learning is to perform fine-tune of a pre-training model. That is, the user selects a successfully trained model from PaddleHub for a new task based on the current task scenario, and the dataset used by the model is similar to the dataset of the new scenario. In this case, you only needs to perform fine-tune of the parameters of the model (**Fine-tune**) during the training of the current task scenario using the data of the new scenario. Transfer learning has attracted many researchers because it is a good solution to the following problems in deep learning:

* In some research areas, there are only a small amount of annotated data, and the cost of data annotation is high, which is not enough to train a sufficiently robust neural network.
* The training of large-scale neural networks relies on large computational resources, which is difficult to implement for a common user.
* Models that address generalized needs do not perform as well as expected for specific applications.

In order to make it easier for developers to apply transfer learning, Paddle has open-sourced PaddleHub – a pre-training model management tool. With just ten lines of codes, developers can complete the transfer learning process. This section describes comprehensive transfer learning by using the PaddleHub.

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