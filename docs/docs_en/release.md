# Release Note

## `v2.3.0`

### [1、Support text-to-image domain model]
  - Add five text-to-image domain models based on disco diffusion, in which three models are for English and two for Chinese. Especially, Chinese text-to-image model [disco_diffusion_ernievil_base](https://aistudio.baidu.com/aistudio/projectdetail/4444998) is based on Baidu **ERNIE-ViL**，welcome to experience.

### 【2、Support Wenxin large models API】
  - Add api call for [**ERNIE-ViLG**](https://aistudio.baidu.com/aistudio/projectdetail/4445016) model, which supports text-to-image task。
  - Add api call for [**ERNIE 3.0 Zeus**](https://aistudio.baidu.com/aistudio/projectdetail/4445054) model, which supports applications such as writing essays, summarization, couplets, question answering, writing novels and completing text.

## `v2.1.0`

### [ 1. Improvements]

- Add supports for five new models, including two high-precision semantic segmentation models based on VOC dataset and three voice classification models.
- Enforce the transfer learning capabilities for image semantic segmentation, text semantic matching and voice classification on related datasets.

### [ 2. Upgrades of deployment capabilities]

- Add the export function APIs for two kinds of model formats, i.,e,  ONNX and PaddleInference.
- **Important Open-Source Ecological Cooperation**: add the support for [BentoML](https://github.com/bentoml/BentoML/), which is  a cloud native framework for serving  deployment. Users can easily serve pre-trained models from [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) by following the [Tutorial notebooks](https:// github.com/PaddlePaddle/PaddleHub/tree/release/v2.1/demo/serving/BentoML).  Also, see this announcement and [Release note](https://github.com/bentoml/BentoML/releases/tag/v0.12.1)  from BentoML. (Many thanks to @[parano](https://github.com/parano) @[cqvu](https://github.com/cqvu) @[deehrlic](https://github.com/deehrlic) for contributing this feature in PaddleHub)

### [ 3. Bug fixes ]

 - [#7da1230](https://github.com/PaddlePaddle/PaddleHub/commit/7da12302dd77e3d739da72821d41715ad8a7c79c) Fixed the problem that the model cannot resume training if metrics is not recorded.
 - [#b0b3144](https://github.com/PaddlePaddle/PaddleHub/commit/b0b3144eff34e47cac8fc450c8b7cb6c557f9b84) Fixed the problem that the thread did not exit normally when the evaluation process was abnormal.
 - [#30aace4](https://github.com/PaddlePaddle/PaddleHub/commit/30aace46414bbeef02beb75b7128f48fada82150) Improve the model installation process.

## `v2.0.0`

* 发布 2.0版本，全面迁移动态图编程模式，模型开发调试更加方便，finetune接口更加灵活易用。
* 视觉类任务迁移学习能力全面升级，支持图像分类、图像着色、风格迁移等多种任务。
* BERT、ERNIE、RoBERTa等Transformer类模型升级至动态图，支持文本分类、序列标注的Fine-Tune能力。
* 新增词向量模型61个，其中包含中文模型51个，英文模型10个。
* 优化服务化部署Serving能力，支持多卡预测、自动负载均衡，性能大幅度提升。
* 新增自动数据增强能力Auto Augment，能高效地搜索适合数据集的数据增强策略组合。

## `v2.0.0-beta1`

* BERT、ERNIE、RoBERTa等Transformer类模型升级至动态图，增加[文本分类](../../demo/text_classification)的Fine-Tune能力
* 修复部分已知问题

## `v2.0.0-beta0`

* 全面迁移动态图编程模式，模型开发调试更加方便，finetune接口更加灵活易用。
* 优化服务化部署Serving能力，支持多卡预测、自动负载均衡，性能大幅度提升。
* 视觉类迁移学习能力全面升级，支持[图像分类](../../demo/image_classification)、[图像着色](../../demo/colorization)、[风格迁移](../../demo/style_transfer)等多种视觉任务。

## `v1.8.1`

* 『[图像分割](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=ImageSegmentation)』新增轻量级人像分割模型Humanseg，支持移动端实时分割
* 增强文本匹配任务性能，使用[EMNLP2019-Sentence-BERT](https://arxiv.org/abs/1908.10084)作为文本匹配任务网络，可同时大幅提升准确率和预测速度。配套教程：[pointwise文本语义匹配](https://aistudio.baidu.com/aistudio/projectdetail/705526)、[pairwise文本语义匹配](https://aistudio.baidu.com/aistudio/projectdetail/709472)
* 修复文本分类选择F1作为评价指标，运行错误

## `v1.8.0`

* 预训练模型丰富，一键完成更多
  * 『[文本生成](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=TextGeneration)』新增基于ERNIE-tiny和ERNIE-gen的对联和写诗生成模型，支持一键自动写诗和对对联。
  *  『[词法分析](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=LexicalAnalysis)』新增jieba的paddle模式切词模型，可一键完成中文分词、关键词抽取等功能。
  * 『[语义表示](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel)』新增基于网页、小说、新闻三类大规模文本数据的LDA主题模型及其语义相似度计算接口。
* Fine-tune API升级，提升灵活性并支持更多任务
   * 新增Tokenizer API，支持更加灵活的切词、切字模式和自定义切词工具拓展。
   * 新增[文本生成](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.8/demo/text_generation)任务，支持Seq2Seq任务的Fine-tuning。
  * 新增文本匹配任务，支持[Pointwise](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.8/demo/pointwise_text_matching)、[Pairwise](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.8/demo/pairwise_text_matching)两种文本匹配训练模式，更便捷完成语义匹配任务。

## `v1.7.0`

* 丰富预训练模型，提升应用性
  * 新增VENUS系列视觉预训练模型[yolov3_darknet53_venus](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_venus&en_category=ObjectDetection)，[faster_rcnn_resnet50_fpn_venus](https://www.paddlepaddle.org.cn/hubdetail?name=faster_rcnn_resnet50_fpn_venus&en_category=ObjectDetection)，可大幅度提升图像分类和目标检测任务的Fine-tune效果
  * 新增工业级短视频分类模型[videotag_tsn_lstm](https://paddlepaddle.org.cn/hubdetail?name=videotag_tsn_lstm&en_category=VideoClassification)，支持3000类中文标签识别
  * 新增轻量级中文OCR模型[chinese_ocr_db_rcnn](https://www.paddlepaddle.org.cn/hubdetail?name=chinese_ocr_db_rcnn&en_category=TextRecognition)、[chinese_text_detection_db](https://www.paddlepaddle.org.cn/hubdetail?name=chinese_text_detection_db&en_category=TextRecognition)，支持一键快速OCR识别
  * 新增行人检测、车辆检测、动物识别、Object等工业级模型

* Fine-tune API升级
  * 文本分类任务新增6个预置网络，包括CNN, BOW, LSTM, BiLSTM, DPCNN等
  * 使用VisualDL可视化训练评估性能数据

## `v1.6.2`

* 修复图像分类在windows下运行错误

## `v1.6.1`

* 修复windows下安装PaddleHub缺失config.json文件

## `v1.6.0`

* NLP Module全面升级，提升应用性和灵活性
  * lac、senta系列(bow、cnn、bilstm、gru、lstm)、simnet_bow、porn_detection系列(cnn、gru、lstm)升级高性能预测，性能提升高达50%
  * ERNIE、BERT、RoBERTa等Transformer类语义模型新增获取预训练embedding接口get_embedding，方便接入下游任务，提升应用性
  * 新增RoBERTa通过模型结构压缩得到的3层Transformer模型[rbt3](https://www.paddlepaddle.org.cn/hubdetail?name=rbt3&en_category=SemanticModel)、[rbtl3](https://www.paddlepaddle.org.cn/hubdetail?name=rbtl3&en_category=SemanticModel)

* Task predict接口增加高性能预测模式accelerate_mode，性能提升高达90%

* PaddleHub Module创建流程开放，支持Fine-tune模型转化，全面提升应用性和灵活性
  * [预训练模型转化为PaddleHub Module教程](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/contribution/contri_pretrained_model.md)
  * [Fine-tune模型转化为PaddleHub Module教程](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/finetuned_model_to_module.md)

* [PaddleHub Serving](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md)优化启动方式，支持更加灵活的参数配置

## `v1.5.2`

* 优化pyramidbox_lite_server_mask、pyramidbox_lite_mobile_mask模型的服务化部署性能

## `v1.5.1`

* 修复加载module缺少cache目录的问题

## `v1.5.0`

* 升级PaddleHub Serving，提升性能和易用性
   * 新增文本Embedding服务[Bert Service](./tutorial/bert_service.md), 轻松获取文本embedding；
      * 代码精短，易于使用。服务端/客户端一行命令即可获取文本embedding；  
      * 更高性能，更高效率。通过Paddle AnalysisPredictor API优化计算图，提升速度减小显存占用
      * 随"机"应变，灵活扩展。根据机器资源和实际需求可灵活增加服务端数量，支持多显卡多模型计算任务
   * 优化并发方式，多核环境中使用多线程并发提高整体QPS

* 优化PaddleHub迁移学习组网Task功能，提升易用性
   * 增加Hook机制，支持[修改Task内置方法](./tutorial/hook.md)
   * 增加colorlog，支持日志彩色显示
   * 改用save_inference_model接口保存模型，方便模型部署
   * 优化predict接口，增加return_result参数，方便用户直接获取预测结果

* 优化PaddleHub Dataset基类，加载[自定义数据](./tutorial/how_to_load_data.md)代码更少、更简单


## `v1.4.1`

* 修复利用Transformer类模型完成序列标注任务适配paddle1.6版本的问题
* Windows下兼容性提升为python >= 3.6

## `v1.4.0`

* 新增预训练模型ERNIE tiny
* 新增数据集：INEWS、BQ、DRCD、CMRC2018、THUCNEWS，支持ChineseGLUE（CLUE）V0 所有任务
* 修复module与PaddlePaddle版本兼容性问题
* 优化Hub Serving启动过程和模型加载流程，提高服务响应速度


## `v1.3.0`

* 新增PaddleHub Serving服务部署
  * 新增[hub serving](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub-Serving%E4%B8%80%E9%94%AE%E6%9C%8D%E5%8A%A1%E9%83%A8%E7%BD%B2)命令，支持一键启动Module预测服务部署
* 新增预训练模型：
  * roberta_wwm_ext_chinese_L-24_H-1024_A-16
  * roberta_wwm_ext_chinese_L-12_H-768_A-12
  * bert_wwm_ext_chinese_L-12_H-768_A-12
  * bert_wwm_chinese_L-12_H-768_A-12
* AutoDL Finetuner优化使用体验
  * 支持通过接口方式回传模型性能
  * 可视化效果优化，支持多trail效果显示

## `v1.2.1`

* 新增**超参优化Auto Fine-tune**，实现给定超参搜索空间，PaddleHub自动给出较佳的超参组合
  * 支持两种超参优化算法：HAZero和PSHE2
  * 支持两种评估方式：FullTrail和PopulationBased
* 新增Fine-tune**优化策略ULMFiT**，包括以下三种设置
  * Slanted triangular learning rates：学习率先线性增加后缓慢降低
  * Discriminative fine-tuning：将计算图划分为n段，不同的段设置不同学习率
  * Gradual unfreezing：根据计算图的拓扑结构逐层unfreezing
* 新增支持用户自定义PaddleHub配置，包括
  * 预训练模型管理服务器地址
  * 日志记录级别
* Fine-tune API升级，灵活性与易用性提升
  * 新增**阅读理解Fine-tune任务**和**回归Fine-tune任务**
  * 新增多指标评测
  * 优化predict接口
  * 可视化工具支持使用tensorboard


## `v1.1.2`

* PaddleHub支持修改预训练模型存放路径${HUB_HOME}


## `v1.1.1`

* PaddleHub支持离线运行
* 修复python2安装PaddleHub失败问题


## `v1.1.0`

* PaddleHub **新增预训练模型ERNIE 2.0**
  * 升级Reader， 支持自动传送数据给Ernie 1.0/2.0
  * 新增数据集GLUE(MRPC、QQP、SST-2、CoLA、QNLI、RTE、MNLI)


## `v1.0.1`

* 安装模型时自动选择与paddlepaddle版本适配的模型


## `v1.0.0`

* 全新发布PaddleHub官网，易用性全面提升
  * 新增网站  https://www.paddlepaddle.org.cn/hub  包含PaddlePaddle生态的预训练模型使用介绍
  * 迁移学习Demo接入AI Studio与AI Book,无需安装即可快速体验

* 新增29个预训练模型，覆盖文本、图像、视频三大领域；目前官方提供40个预训练模型
  * CV预训练模型：
    * 新增图像分类预训练模型11个：SE_ResNeXt, GoogleNet, ShuffleNet等
    * 新增目标检测模型Faster-RCNN和YOLOv3
    * 新增图像生成模型CycleGAN
    * 新增人脸检测模型Pyramidbox
    * 新增视频分类模型4个: TSN, TSM, StNet, Non-Local
  * NLP预训练模型
    * 新增语义模型ELMo
    * 新增情感分析模型5个: Senta-BOW, Senta-CNN, Senta-GRNN, , Senta-LSTM, EmoTect
    * 新增中文语义相似度分析模型SimNet
    * 升级LAC词法分析模型，新增词典干预功能，支持用户自定义分词
* Fine-tune API升级，灵活性与性能全面提升
  * 支持多卡并行、PyReader多线程IO，Fine-tune速度提升60%
  * 简化finetune、evaluate、predict等使用逻辑，提升易用性
  * 增加事件回调功能，方便用户快速实现自定义迁移学习任务
  * 新增多标签分类Fine-tune任务


## `v0.5.0`

正式发布PaddleHub预训练模型管理工具，旨在帮助用户更高效的管理模型并开展迁移学习的工作。

**预训练模型管理**: 通过hub命令行可完成PaddlePaddle生态的预训练模型下载、搜索、版本管理等功能。

**命令行一键使用**: 无需代码，通过命令行即可直接使用预训练模型进行预测，快速调研训练模型效果。目前版本支持以下模型：词法分析LAC；情感分析Senta；目标检测SSD；图像分类ResNet, MobileNet, NASNet等。

**迁移学习**: 提供了基于预训练模型的Fine-tune API，用户通过少量代码即可完成迁移学习，包括BERT/ERNIE文本分类、序列标注、图像分类迁移等。
