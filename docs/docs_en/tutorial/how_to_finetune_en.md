## **Transfer Learning**

### **Overview**

Transfer Learning is a subfield of deep learning that aims to use similarities in data, tasks, or models to transfer knowledge learned in old fields to new fields. In other words, transfer learning refers to the use of existing knowledge to learn new knowledge. For example, people who have learned to ride a bicycle can learn to ride an electric bike faster. A common method of transfer learning is to perform fine-tune of a pre-training model. That is, the user selects a successfully trained model from PaddleHub for a new task based on the current task scenario, and the dataset used by the model is similar to the dataset of the new scenario. In this case, you only needs to perform fine-tune of the parameters of the model (**Fine-tune**) during the training of the current task scenario using the data of the new scenario. Transfer learning has attracted many researchers because it is a good solution to the following problems in deep learning:

* In some research areas, there are only a small amount of annotated data, and the cost of data annotation is high, which is not enough to train a sufficiently robust neural network.
* The training of large-scale neural networks relies on large computational resources, which is difficult to implement for a common user.
* Models that address generalized needs do not perform as well as expected for specific applications.

In order to make it easier for developers to apply transfer learning, Paddle has open-sourced PaddleHub – a pre-training model management tool. With just ten lines of codes, developers can complete the transfer learning process. This section describes comprehensive transfer learning by using the PaddleHub.

### **Prerequisites**

Before starting transfer learning, users need to do the following:

* The user has installed PaddleHub.
* To prepare the data for transfer learning, users can choose to use the dataset provided by PaddleHub or a custom dataset. For details of the custom data, refer to the "Custom dataset Fine-tune" to process the dataset.
* Run the hub install command to install or update the modules used for training. For example, for the ERNIE model, the command format is as follows: You may have already installed the relevant pre-training model in a previous task. However, it is still recommended that you perform this step before you start training, so that you can ensure that the pre-training model is the latest version.

```python
$ hub install ernie==1.2.0
```

### **Transfer Learning Process**

Before you can complete the transfer learning process, you need to write scripts for transfer learning. The scripting process is very simple, requiring only about ten lines of codes. The entire scripting process has the following steps:

1. Import the necessary packages.
2. Load a pre-training model, that is, load the pre-training model provided by PaddleHub.
3. Load Dataset. You can choose to load the dataset that comes with PaddleHub by using the dataset API or write your own class to load the dataset to load the custom dataset.
4. Configure the reader to pre-process the data in dataset. Organize in a specific format and input to the model for training.
5. Choose an optimization strategy that includes various pre-training parameters, such as, what learning rate variation strategy, what type of optimizer, and what type of regularization.
6. Set the RunConfig. It contains a number of training-related configurations, including whether  to use the GPU, the number of training rounds (Epoch), training batch size (batch\_size), and so on.
7. Create a training task. A transfer learning training task contains the task-related Program, Reader and RunConfig set above.
8. Start the Fine-tune. Use the Fine-tune\_and\_eval function to complete the training and evaluation.

### **Learn to Write Transfer Learning Training Scripts**

PaddleHub provides a Finetune API and a pre-training model to perform transfer learning for a variety of different task scenarios, including image classification, text classification, multi-label classification, sequence annotation, retrieval quiz task, regression task, sentence semantic similarity computation, reading comprehension task, and so on. This section describes how to write transfer learning scripts by taking the text classification as an example.

#### **1\. Import the necessary packages.**

```python
import paddlehub as hub  
```

#### **2\. Load the pre-training model.**

The following code is used to load the pre-training model. In this case, the ERNIE pre-training model is used to complete the text classification task. The Enhanced Representation through Knowledge Integration (ERNIE) is a semantic representation model proposed by Baidu, with Transformer Encoder as the basic component. The pre-training process uses richer semantic knowledge and more semantic tasks. Users can use the pre-training model to gradually introduce different custom tasks at any time, such as, naming entity prediction, semantic paragraph relationship recognition, sentence order prediction task, sentiment analysis, and so on.

```python
module = hub.Module(name="ernie")
```

PaddleHub also provides many other pre-training models for transfer learning. On the PaddleHub website, the pre-training models under Image Classification, Semantic Model, and Sentiment Analysis all support transfer learning. Users only need to replace the name value with the name of the pre-training model, for example, the red box on the right.

![](../imgs/Howtofinetune1.png)

#### **3\. Load the dataset.**

After loading the pre-training model, we load the dataset. The sources of datasets used for transfer learning can be divided into customized datasets and datasets provided by PaddleHub. Loading methods vary with datasets.

##### **Load PaddleHub's own datasets.**

If you are using PaddleHub's own dataset, you can use PaddleHub's dataset API to write a single line of code to load the dataset.

```python
dataset = hub.dataset.ChnSentiCorp()
```

ChnSentiCorp is a Chinese sentiment analysis dataset, and aims to determine the sentiment attitude of a paragraph of texts. For example, if the text is "The food is delicious", the corresponding label is "1", indicating a positive evaluation; or if the text is "The room is too small", the corresponding label is "0", indicating a negative evaluation. PaddleHub also provides other text classification datasets. You can choose the corresponding API to replace the values of the dataset in the above code, as listed in the table below.

|       Dataset        |                             Name                             |              API              |
| :------------------: | :----------------------------------------------------------: | :---------------------------: |
|     ChnSentiCorp     |              Chinese Sentiment Analysis Dataset              |  hub.dataset.ChnSentiCorp()   |
|        LCQMC         | A question-and-answer matching Chinese dataset constructed by Harbin Institute of Technology at the International Summit on Natural Language Processing COLING2018, with the goal of determining whether the semantics of two questions are the same. |      hub.dataset.LCQMC()      |
|      NLPCC-DPQA      | The goal of the evaluation task dataset held by the International Natural Language Processing and Chinese Computing Conference NLPCC in 2016 is to select answers that can answer the questions. |   hub.dataset.NLPCC_DPQA()    |
|       MSRA-NER       | A dataset released by Microsoft Asian Research Institute that aims to identify named entities |    hub.dataset.MSRA-NER()     |
|        Toxic         |           English multi-labeled taxonomic datasets           |      hub.dataset.Toxic()      |
|        SQUAD         |            English reading comprehension dataset             |      hub.dataset.SQUAD()      |
|      GLUE-CoLA       |               Text Classification Task Dataset               |   hub.dataset.GLUE("CoLA")    |
|      GLUE-SST-2      |               Sentiment Analysis Task Dataset                |   hub.dataset.GLUE("SST-2")   |
|      GLUE-MNLI       |                 Text Reasoning Task Dataset                  |  hub.dataset.GLUE("MNLI_m")   |
|       GLUE-QQP       |          Sentence pairs Classification Task Dataset          |    hub.dataset.GLUE("QQP")    |
|      GLUE-QNLI       |                Problem Inference Task Dataset                |   hub.dataset.GLUE("QNLI")    |
|      GLUE-STS-B      |                   Regression Task Dataset                    |   hub.dataset.GLUE("STS-B")   |
|      GLUE-MRPC       |          Sentence pairs Classification Task Dataset          |   hub.dataset.GLUE("MRPC")    |
|       GLUE-RTE       |                Text implication task dataset                 |    hub.dataset.GLUE("RTE")    |
|         XNLI         |      Cross-language natural language inference dataset       | hub.dataset.XNLI(language=zh) |
|  ChineseGLUE-TNEWS   | Today's Headlines Chinese News (Short text) Classified Dataset |      hub.dataset.TNews()      |
|  ChineseGLUE-INEWS   |           Internet Sentiment Analysis Task Dataset           |      hub.dataset.INews()      |
|         DRCD         | Delta Reading Comprehension Dataset, a General Domain Traditional Chinese Character Machine Reading Comprehension Dataset |      hub.dataset.DRCD()       |
|       CMRC2018       | Span Extraction Dataset for Chinese Machine Reading Comprehension |    hub.dataset.CMRC2018()     |
|    ChinesGLUE-BQ     | Intelligent Customer Service Chinese Question Matching Dataset |       hub.dataset.BQ()        |
| ChineseGLUE-IFLYTEK  | Chinese long-text classification dataset with over 17,000 text markups on app |     hub.dataset.IFLYTEK()     |
| ChineseGLUE-THUCNEWS | Chinese long text classification dataset, with more than 40,000 Chinese news long text markup data, with 14 categories in total. |    hub.dataset.THUCNEWS()     |
|    DogCatDataset     |        Dataset provided by Kaggle for image binning.         |  hub.dataset.DogCatDataset()  |
|       Food101        | Food photo dataset provided by Kaggle, containing 101 categories |     hub.dataset.Food101()     |
|       Indoor67       | A dataset of 67 indoor scenes released by MIT with the goal of identifying the scene category of an indoor image. |    hub.dataset.Indoor67()     |
|       Flowers        | Flower dataset. There are 5 types of datasets, including "roses", "tulips", "daisy", "sunflowers", and "dandelion". |     hub.dataset.Flowers()     |
|     StanfordDogs     | Dataset released by Stanford University, containing 120 species of dogs for image classification. |  hub.dataset.StanfordDogs()   |

##### **Load custom datasets**

* Load the text class custom dataset. The user only needs to inherit the base class BaseNLPDatast and modify the dataset address and category. Refer to the following codes.

```python
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
# Define Class
class DemoDataset(BaseNLPDataset):
    def __init__(self):
        # path of dateset
        self.dataset_dir = "path/to/dataset"
        super(DemoDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.tsv", # training dataset
            dev_file="dev.tsv",    # dev dataset
            test_file="test.tsv",   # test dataset
            predict_file="predict.tsv",
            train_file_with_header=True,  
            dev_file_with_header=True,  
            test_file_with_header=True,  
            predict_file_with_header=True,
            # Dataset Label
            label_list=["0", "1"])
# Create Dataset Object
dataset = DemoDataset()
```

Then you can get the custom dataset by DemoDataset(). With the data preprocessor and pre-training models such as ERNIE, you can complete the text class task.

* Load the image class custom dataset. The user only needs to inherit the base class BaseCVDatast and modify the dataset storage address. Refer to the following codes.

```python
from paddlehub.dataset.base_cv_dataset import BaseCVDataset

class DemoDataset(BaseCVDataset):
   def __init__(self):
       # Path of dataset
       self.dataset_dir = "/test/data"
       super(DemoDataset, self).__init__(
           base_path=self.dataset_dir,
           train_list_file="train_list.txt",  
           validate_list_file="validate_list.txt",
           test_list_file="test_list.txt",  
           predict_file="predict_list.txt",  
           label_list_file="label_list.txt",  
           )
dataset = DemoDataset()
```

Then you can get the custom dataset by DemoDataset(). With the data preprocessor and pre-training models, you can complete the vision class transfer learning task.

#### **4\. Configure the Data Preprocessor.**

Read data of NLP or CV dataset by using PaddleHub's Data Preprocessor API.

```python
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),  
    max_seq_len=128,  
    sp_model_path=module.get_spm_path(),  
    word_dict_path=module.get_word_dict_path())  

```

For different task types, users can select different Readers.

|        Data Readers        |                         Description                          | Task Type | API Examples                                                 |
| :------------------------: | :----------------------------------------------------------: | :-------: | :----------------------------------------------------------- |
|       ClassifyReader       | It is the data preprocessor suitable for the Transformer pre-training model (ERNIE/BERT). |    NLP    | reader = hub.reader.ClassifyReader(     dataset=dataset,     vocab_path=module.get_vocab_path(),     max_seq_len=128,     sp_model_path=module.get_spm_path(),     word_dict_path=module.get_word_dict_path()) |
|     LACClassifyReader      | A data preprocessor using the LAC module as a word cutter, suitable for tasks such as Senta and ELMo that require granularity-based word segmentation. |    NLP    | reader = hub.reader.LACClassifyReader(     dataset=dataset,     vocab_path=module.get_vocab_path()) |
|    SequenceLabelReader     | Sequence annotation preprocessor for the Transformer class model (ERNIE/BERT). |    NLP    | reader = hub.reader.SequenceLabelReader(     dataset=dataset,     vocab_path=module.get_vocab_path(),     max_seq_len=128,     sp_model_path=module.get_spm_path(),     word_dict_path=module.get_word_dict_path()) |
|  MultiLabelClassifyReader  | Multi-label classification preprocessor for the Transformer class model (ERNIE/BERT). |    NLP    | reader = hub.reader.MultiLabelClassifyReader(     dataset=dataset,     vocab_path=module.get_vocab_path(),     max_seq_len=128) |
| ReadingComprehensionReader | Reading comprehension task preprocessor for the Transformer class model (ERNIE/BERT). |    NLP    | reader = hub.reader.ReadingComprehensionReader(     dataset=dataset,     vocab_path=module.get_vocab_path(),     max_seq_length=384) |
|      RegressionReader      |      A data preprocessor suitable for regression tasks.      |    NLP    | reader = hub.reader.RegressionReader(     dataset=dataset,     vocab_path=module.get_vocab_path(),     max_seq_len=args.max_seq_len) |
| ImageClassificationReader  | Preprocessor suitable for image classification data. Modify the size of the input image. Perform standardization, image broadening, and other operations. |    CV     | reader = hub.reader.ImageClassificationReader(     image_width=module.get_expected_image_width(),     image_height=module.get_expected_image_height(),     images_mean=module.get_pretrained_images_mean(),     images_std=module.get_pretrained_images_std(),     dataset=dataset) |

#### **5\. Choose an Optimization Strategy.**

In PaddleHub, the Strategy class encapsulates a set of fine-tuning strategies for transfer learning. Strategies include what learning rate variation strategy for pre-training parameters, what type of optimizer, what type of regularization, and so on. In the text classification task, we use AdamWeightDecayStrategy optimization strategy. Refer to the following code.

```python
strategy = hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,   # Fine-tune max learning rate
    weight_decay=0.01,    # defaule 0.01
    warmup_proportion=0.1,  #warmup_proportion>0, for example: 0.1
    lr_scheduler="linear_decay",  
)
```

PaddleHub also provides a variety of APIs for various optimization strategies, in addition to AdamWeightDecayStrategy.

| Optimization Strategies | Description                                                  | API Examples                                                 |
| :---------------------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
| DefaultFinetuneStrategy | The default optimization strategy. The corresponding parameters are as follows: <br/>\* learning\_rate: The global learning rate. The default value is 1e-4. <br/>\* optimizer\_name: Optimizer\_name: The default is adam. <br/>\* regularization\_coeff: The regularization λ parameter. The default value is 1e-3. <br/> This optimization strategy is recommended for the image classification task. | strategy = hub.DefaultFinetuneStrategy(     learning_rate=1e-4,     optimizer_name="adam",     regularization_coeff=1e-3) |
| AdamWeightDecayStrategy | A learning rate decay strategy based on Adam optimizer. The corresponding parameters are as follows: <br/>\* learning\_rate: Global learning rate. The default value is 1e-4. <br/>\* lr\_scheduler: Learning rate scheduling method. The default value is "linear\_decay". <br/>\* warmup\_proportion: Warmup proportion. <br/>\* weight\_decay: Learning rate decay rate. <br/>\* optimizer\_name: Optimizer name. The default value is adam.<br/> This optimization strategy is recommended for text classification, reading comprehension, and other tasks. | strategy = hub.AdamWeightDecayStrategy(     learning_rate=1e-4,     lr_scheduler="linear_decay",     warmup_proportion=0.0,     weight_decay=0.01,     optimizer_name="adam") |
|  L2SPFinetuneStrategy   | Finetune strategy using the L2SP regular as the penalty factor. The corresponding parameters are as follows: <br/>\* learning\_rate: The global learning rate. The default value is 1e-4. <br/>\* optimizer\_name: Optimizer\_name: The default is adam. <br/>\* regularization\_coeff: The regularization λ parameter. The default value is 1e-3. | strategy = hub.L2SPFinetuneStrategy(     learning_rate=1e-4,     optimizer_name="adam",     regularization_coeff=1e-3) |
|     ULMFiTStrategy      | The strategy implements the three strategies proposed in the ULMFiT paper: <br/> \* Slanted triangular learning rates: a strategy of learning to rise and then fall. <br/>\* Discriminative fine-tuning: a strategy of decreasing learning rates layer by layer. It can slow down the underlying update rate. <br/>\* Gradual unfreezing: a layer-by-layer unfreezing strategy. It gives a priority to updating the upper layer and then slowly unfreezes the lower layer, to participate in the update. <br/> The corresponding parameters are as follows: <br/> \* learning\_rate: The global learning rate. The default value is 1e-4. <br/> \* optimizer\_name: Optimizer\_name: The default value is adam. <br/> \* cut\_fraction: Sets the ratio of number of steps of learning rate increase of “Slanted triangular learning rates” to the total number of training steps. The default value is 0.1. If it is set to 0, “Slanted triangular learning rates” is not used. <br/> \* ratio: Sets the ratio of the decrease minimum learning rate to the increase maximum learning rate of “Slanted triangular learning rates”. The default value is 32, indicating that the minimum learning rate is 1/32 of the maximum learning rate. <br/> \* dis\_blocks: Sets the number of blocks in the Discriminative fine-tuning. The default value is 3. If it is set to 0, the Discriminative fine-tuning is not used. <br/> \* factor: Sets the decay rate of the Discriminative fine-tuning. The default value is 2.6, indicating that the learning rate of the next layer is 1/2.6 of the upper layer. <br/> \* frz\_blocks: Sets the number of blocks in the gradual unfreezing. The concept of block is the same as that in "dis\_blocks". | strategy = hub.ULMFiTStrategy(     learning_rate=1e-4,     optimizer_name="adam",     cut_fraction=0.1,     ratio=32,     dis_blocks=3,     factor=2.6,     frz_blocks=3) |

#### **6\. Set the RunConfig.**

In PaddleHub, users can use RunConfig in the Finetune API to configure the parameters when a Task is in Fine-tune, including Epochs, batch size, and whether to use GPU for training. The example codes are as follows:

```python
config = hub.RunConfig(use_cuda=True, num_epoch=3, batch_size=32, strategy=strategy)
```

#### **7\. Assemble the Training Task.**

Once we have a suitable pre-training model and load the dataset to be transferred, we start to assemble a Task. In PaddleHub, the Task represents a fine-tuned task. The Task contains the information related to the execution of the task, such as Program, Reader, and RunConfig. The task description corresponding to the [TextClassifierTask](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/reference/task/text_classify_task.md) can be found here. The specific implementation solution is as follows:

1. Get the context of the module (PaddleHub's pre-training model), including the input and output variables, and Paddle Program (executable model format).
2. Find the feature\_map of the extraction layer from the output variables of the pre-training model, and access a full connection layer behind the feature\_map. In the following codes, it is specified by the pooled\_output parameter of the hub.TextClassifierTask.
3. The input layer of the network remains unchanged, still starting from the input layer, as specified in the following code via the feed\_list variable of the hub.TextClassifierTask. The hub.TextClassifierTask specifies our requirements for intercepting the model network with these two parameters. According to this configuration, the network we intercept is from the input layer to the last layer of feature extraction "pooled\_output", indicating that we will use the intercepted Network for transfer learning training.

```python
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)  
pooled_output = outputs["pooled_output"]

feed_list = [  
    inputs["input_ids"].name,  
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.TextClassifierTask(
    data_reader=reader,  
    feature=pooled_output,  
    feed_list=feed_list,  
    num_classes=dataset.num_labels,  
    metrics_choices = ["acc"],
    config=config)  
```

PaddleHub comes with pre-built Tasks for common tasks. Each task has a specific application scenario and corresponding metrics to meet the different user requirements.

|        Task Type         |                         Description                          |            Task Type            |
| :----------------------: | :----------------------------------------------------------: | :-----------------------------: |
|   ImageClassifierTask    | The Task is added with one or more full connection layers to create a classification task for Fine-tune based on the input features. The metric is the accuracy, and the loss function is the cross-entropy Loss. |    Image classification task    |
|    TextClassifierTask    | The Task is added with a Dropout layer and one or more full connection layers to create a text classification task for fine-tune based on the input features. The metric is accuracy, and the loss function is cross-entropy Loss. |    Text classification task     |
|    SequenceLabelTask     | The Task is added with one full connection layer or a full connection layer and CRF layer to create a sequence label task for Fine-tune based on the input features. The metric is F1, and the loss function is the cross-entropy Loss. |    Sequence Annotation Task     |
| MultiLabelClassifierTask | The Task is added with one or more full connection layers to create a multi-label classification task for fine-tune based on the input features. The metric is average AUC of multiple labels, and the loss function is the average cross-entropy of multiple labels. | Multi-label classification task |
|      RegressionTask      | The Task is added with a Dropout layer and one or more full connection layers to create a text regression task for fine-tune based on the input features. The metric is accuracy, and the loss function is the  mean variance loss function. |      Text Regression Task       |
| ReadingComprehensionTask | The Task is added with one full connection layer to create a reading comprehension task for fine-tune based on the input features. The loss function is the cross-entropy Loss. |   Reading Comprehension Task    |

Before setting each Task, the user needs to know the input and output of the pre-training model of the transfer learning, that is, "feed\_list" and "pooled\_output" in the above codes. The specific input and output codes can be divided into the following:

* Image classification model

```
input_dict, output_dict, program = module.context(trainable=True)
feature_map = output_dict["feature_map"]
feed_list = [input_dict["image"].name]
```

* Natural language processing model (excluding word2vec\_skipgram, simnet\_bow, text matching, and text generation models).

```
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)  
pooled_output = outputs["pooled_output"]
feed_list = [  
    inputs["input_ids"].name,  
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]
```

* word2vec\_skipgram model

```
inputs, outputs, program = module.context(trainable=True)
word_ids = inputs["word_ids"]
embedding = outputs["word_embs"]
```

* simnet\_bow model

```
inputs, outputs, program = module.context(trainable=True, max_seq_len=args.max_seq_len, num_slots=2)
query = outputs["emb"]
title = outputs['emb_2']
```

* Pairwise Text Matching Model

```
inputs, outputs, program = module.context(trainable=True, max_seq_len=args.max_seq_len, num_slots=3)
query = outputs["emb"]
left = outputs['emb_2']
right = outputs['emb_3']
```

* Pointwise Text Matching

```
inputs, outputs, program = module.context(trainable=True, max_seq_len=args.max_seq_len, num_slots=2)
query = outputs["emb"]
title = outputs['emb_2']
```

* Text Generation Model

```
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)
pooled_output = outputs["pooled_output"]
sequence_output = outputs["sequence_output"]
```

#### **8\. Start the Fine-tune. Use the Finetune\_and\_eval Function to Complete the Training and Evaluation.**

```python
cls_task.finetune_and_eval()
```

The information is displayed as follows: You can see the evaluation results of the training, Loss value, accuracy, and so on.

```
[2020-07-28 21:28:21,658] [   TRAIN] - step 810 / 900: loss=0.05022 acc=0.97813 [step/sec: 4.07]
[2020-07-28 21:28:24,115] [   TRAIN] - step 820 / 900: loss=0.04719 acc=0.98125 [step/sec: 4.07]
[2020-07-28 21:28:26,574] [   TRAIN] - step 830 / 900: loss=0.06895 acc=0.98125 [step/sec: 4.07]
[2020-07-28 21:28:29,035] [   TRAIN] - step 840 / 900: loss=0.07830 acc=0.97813 [step/sec: 4.07]
[2020-07-28 21:28:31,490] [   TRAIN] - step 850 / 900: loss=0.07279 acc=0.97500 [step/sec: 4.08]
[2020-07-28 21:28:33,939] [   TRAIN] - step 860 / 900: loss=0.03220 acc=0.99375 [step/sec: 4.09]
[2020-07-28 21:28:36,388] [   TRAIN] - step 870 / 900: loss=0.05016 acc=0.98750 [step/sec: 4.09]
[2020-07-28 21:28:38,840] [   TRAIN] - step 880 / 900: loss=0.05604 acc=0.98750 [step/sec: 4.08]
[2020-07-28 21:28:41,293] [   TRAIN] - step 890 / 900: loss=0.05622 acc=0.98125 [step/sec: 4.08]
[2020-07-28 21:28:43,748] [   TRAIN] - step 900 / 900: loss=0.06642 acc=0.97813 [step/sec: 4.08]
[2020-07-28 21:28:43,750] [    INFO] - Evaluation on dev dataset start
[2020-07-28 21:28:46,654] [    EVAL] - [dev dataset evaluation result] loss=0.17890 acc=0.94079 [step/sec: 13.23]
[2020-07-28 21:28:46,657] [    INFO] - Evaluation on dev dataset start
[2020-07-28 21:28:49,527] [    EVAL] - [dev dataset evaluation result] loss=0.17890 acc=0.94079 [step/sec: 13.39]
[2020-07-28 21:28:49,529] [    INFO] - Load the best model from ckpt_20200728212416/best_model
[2020-07-28 21:28:50,112] [    INFO] - Evaluation on test dataset start
[2020-07-28 21:28:52,987] [    EVAL] - [test dataset evaluation result] loss=0.14264 acc=0.94819 [step/sec: 13.36]
[2020-07-28 21:28:52,988] [    INFO] - Saving model checkpoint to ckpt_20200728212416/step_900
[2020-07-28 21:28:55,789] [    INFO] - PaddleHub finetune finished.
```

After completing the model training with Fine-tune, PaddleHub will automatically save the best model in the validation set in the corresponding ckpt directory (CKPT\_DIR). The user can make predictions with reference to the following codes, where the inferred label value 0 indicates a negative evaluation, and 1 indicates a positive evaluation.

```python
import numpy as np


# 待预测数据
data = [
    ["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"],
    ["交通方便；环境很好；服务态度很好 房间较小"],
    ["19天硬盘就罢工了，算上运来的一周都没用上15天，可就是不能换了。唉，你说这算什么事呀！"]
]

index = 0
run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
for batch_result in results:
    # 预测类别取最大分类概率值
    batch_result = np.argmax(batch_result[0], axis=1)
    for result in batch_result:
        print("%s\tpredict=%s" % (data[index][0], result))
        index += 1
```

The prediction results are as follows:

```
[2020-07-28 18:06:45,441] [    INFO] - PaddleHub predict start
[2020-07-28 18:06:45,442] [    INFO] - The best model has been loaded
[2020-07-28 18:06:48,406] [    INFO] - PaddleHub predict finished.

这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般	predict=0
交通方便；环境很好；服务态度很好 房间较小	predict=1
19天硬盘就罢工了，算上运来的一周都没用上15天，可就是不能换了。唉，你说这算什么事呀！	predict=0
```
