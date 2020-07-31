PaddlePaddle 提供了丰富的模型，使得用户可以采用模块化的方法解决各种学习问题。本文，我们将整体介绍PaddleHub中已经准备好的丰富的预训练模型。

* 如果是想了解具体预训练模型的使用可以继续学习本课程，也可以参考 [PaddleHub预训练模型库](https://www.paddlepaddle.org.cn/hublist)。

* 如果想了解更多模型组网网络结构源代码请参考 [飞桨模型库](https://github.com/PaddlePaddle/models)。

## PaddleHub预训练模型库
* [飞桨优势特色模型](#飞桨优势特色模型)
* [图像](#图像)
  * [图像分类](#图像分类)
  * [目标检测](#目标检测)
  * [图像分割](#图像分割)
  * [关键点检测](#关键点检测)
  * [图像生成](#图像生成)
* [文本](#文本)
  * [中文词法分析与词向量](#中文词法分析与词向量)
  * [情感分析](#情感分析)
  * [文本相似度计算](#文本相似度计算)
  * [语义表示](#语义表示)
* [视频](#视频)



## 百度飞桨独有优势特色模型

|            | **模型名称**                                                 | **Master模型推荐辞**                                       |
| ---------- | :----------------------------------------------------------- | ---------------------------------------------------------- |
| 图像分类 | [菜品识别](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_dishes&en_category=ImageClassification) | 私有数据集训练，支持8416种菜品的分类识别，适合进一步菜品方向微调 |
| 图像分类 | [动物识别](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification) | 私有数据集训练，支持7978种动物的分类识别，适合进一步动物方向微调 |
| 目标检测   | [YOLOv3](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_coco2017&en_category=ObjectDetection) | 实现精度相比原作者**提高5.9 个绝对百分点**，性能极致优化。 |
| 人脸检测 | [人脸检测](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_server&en_category=FaceDetection) | 百度自研，18年3月WIDER Face 数据集**冠军模型**，           |
| 人脸检测 | [口罩人脸检测与识别](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_server_mask&en_category=FaceDetection) | 业界**首个开源口罩人脸检测与识别模型**，引起广泛关注。     |
| 目标检测 | [行人检测](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_pedestrian&en_category=ObjectDetection) | 百度自研模型，海量私有数据集训练，可以应用于智能视频监控，人体行为分析，客流统计系统，智能交通等领域 |
| 目标检测 | [车辆检测](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_vehicles&en_category=ObjectDetection) | 百度自研模型，支持car (汽车)，truck (卡车)，bus (公交车)，motorbike (摩托车)，tricycle (三轮车)等车型的识别 |
| 语义分割   | [人像分割](https://www.paddlepaddle.org.cn/hubdetail?name=deeplabv3p_xception65_humanseg&en_category=ImageSegmentation) | 百度**自建数据集**训练，人像分割效果卓越。                 |
| 语义分割   | [人体解析](https://www.paddlepaddle.org.cn/hubdetail?name=ace2p&en_category=ImageSegmentation) | CVPR2019 LIP挑战赛中**满贯三冠王**。人体解析任务必选。     |
| 语义分割   | [肺炎CT影像分析](https://www.paddlepaddle.org.cn/hubdetail?name=Pneumonia_CT_LKM_PP&en_category=ImageSegmentation) | 助力连心医疗开源**业界首个**肺炎CT影像分析模型             |
| GAN        | [风格迁移](https://www.paddlepaddle.org.cn/hubdetail?name=stylepro_artistic&en_category=GANs) | 百度自研风格迁移模型，趣味模型，**推荐尝试**                         |
| OCR | [超轻量中英文OCR文字识别](https://www.paddlepaddle.org.cn/hubdetail?name=chinese_ocr_db_crnn_mobile&en_category=TextRecognition) | 业界开源最小，8.6M超轻量中英文识别模型。支持中英文识别；支持倾斜、竖排等多种方向文字识别，**强力推荐** |
| 视频分类 | [超大规模视频分类](https://www.paddlepaddle.org.cn/hubdetail?name=videotag_tsn_lstm&en_category=VideoClassification) | 百度自研模型，基于千万短视频预训练的视频分类模型，可直接预测短视频的中文标签 |
| 词法分析   | [LAC ](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis) | 百度**自研中文特色**模型词法分析任务。                     |
| 情感分析   | [Senta](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis) | 百度自研情感分析模型，海量中文数据训练。                   |
| 情绪识别   | [emotion_detection](https://www.paddlepaddle.org.cn/hubdetail?name=emotion_detection_textcnn&en_category=SentimentAnalysis) | 百度自研对话识别模型，海量中文数据训练。                   |
| 文本相似度 | [simnet](https://www.paddlepaddle.org.cn/hubdetail?name=simnet_bow&en_category=SemanticModel) | 百度自研短文本相似度模型，海量中文数据训练。               |
| 文本审核   | [porn_detection](https://www.paddlepaddle.org.cn/hubdetail?name=porn_detection_gru&en_category=TextCensorship) | 百度自研色情文本审核模型，海量中文数据训练。               |
| 语义模型   | [ERNIE](https://www.paddlepaddle.org.cn/hubdetail?name=ERNIE&en_category=SemanticModel) | **SOTA语义模型，中文任务全面优于BERT**。               |
| 语音合成   | WaveFlow（即将开源）                                         | 百度自研模型，海量私有数据集训练                 |

## 图像

#### 图像分类

图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

**注：** **如果你是资深开发者，那可以随意按需使用**，**假如你是新手，服务器端优先选择Resnet50，移动端优先选择MobileNetV2**

| **模型名称** | **模型简介** |
| - | - |
| [AlexNet](https://www.paddlepaddle.org.cn/hubdetail?name=alexnet_imagenet&en_category=ImageClassification) | 首次在 CNN 中成功的应用了 ReLU, Dropout 和 LRN，并使用 GPU 进行运算加速 |
| [VGG19](https://www.paddlepaddle.org.cn/hubdetail?name=vgg19_imagenet&en_category=ImageClassification) | 在 AlexNet 的基础上使用 3*3 小卷积核，增加网络深度，具有很好的泛化能力 |
| [GoogLeNet](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/image_classification) | 在不增加计算负载的前提下增加了网络的深度和宽度，性能更加优越 |
| [ResNet50](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_50_imagenet&en_category=ImageClassification) | Residual Network，引入了新的残差结构，解决了随着网络加深，准确率下降的问题 |
| [Inceptionv4](https://www.paddlepaddle.org.cn/hubdetail?name=inception_v4_imagenet&en_category=ImageClassification) | 将 Inception 模块与 Residual Connection 进行结合，通过ResNet的结构极大地加速训练并获得性能的提升 |
| [MobileNetV2](https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v2_imagenet&en_category=ImageClassification) | MobileNet结构的微调，直接在 thinner 的 bottleneck层上进行 skip learning 连接以及对 bottleneck layer 不进行 ReLu 非线性处理可取得更好的结果 |
| [se_resnext50](https://www.paddlepaddle.org.cn/hubdetail?name=se_resnext50_32x4d_imagenet&en_category=ImageClassification) | 在ResNeXt 基础、上加入了 SE(Sequeeze-and-Excitation) 模块，提高了识别准确率，在 ILSVRC 2017 的分类项目中取得了第一名 |
| [ShuffleNetV2](https://www.paddlepaddle.org.cn/hubdetail?name=shufflenet_v2_imagenet&en_category=ImageClassification) | ECCV2018，轻量级 CNN 网络，在速度和准确度之间做了很好地平衡。在同等复杂度下，比 ShuffleNet 和 MobileNetv2 更准确，更适合移动端以及无人车领域 |
| [efficientNetb7](https://www.paddlepaddle.org.cn/hubdetail?name=efficientnetb7_imagenet&en_category=ImageClassification) | 同时对模型的分辨率，通道数和深度进行缩放，用极少的参数就可以达到SOTA的精度。 |
| [xception71](https://www.paddlepaddle.org.cn/hubdetail?name=xception71_imagenet&en_category=ImageClassification) | 对inception-v3的改进，用深度可分离卷积代替普通卷积，降低参数量同时提高了精度。 |
| [dpn107](https://www.paddlepaddle.org.cn/hubdetail?name=dpn107_imagenet&en_category=ImageClassification) | 融合了densenet和resnext的特点。 |
| [DarkNet53](https://www.paddlepaddle.org.cn/hubdetail?name=darknet53_imagenet&en_category=ImageClassification) | 检测框架yolov3使用的backbone，在分类和检测任务上都有不错表现。 |
| [DenseNet161](https://www.paddlepaddle.org.cn/hubdetail?name=densenet161_imagenet&en_category=ImageClassification) | 提出了密集连接的网络结构，更加有利于信息流的传递。 |
| [ResNeXt152_vd](https://www.paddlepaddle.org.cn/hubdetail?name=resnext152_64x4d_imagenet&en_category=ImageClassification) | 提出了cardinatity的概念，用于作为模型复杂度的另外一个度量，有效地提升模型精度。 |



#### 目标检测

目标检测任务的目标是给定一张图像或是一个视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。对于计算机而言，能够“看到”的是图像被编码之后的数字，但很难解图像或是视频帧中出现了人或是物体这样的高层语义概念，也就更加难以定位目标出现在图像中哪个区域。目标检测模型请参考 (https://github.com/PaddlePaddle/PaddleDetection)

| 模型名称                                                     | 模型简介                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [SSD](https://www.paddlepaddle.org.cn/hubdetail?name=ssd_mobilenet_v1_pascal&en_category=ObjectDetection) | 很好的继承了 MobileNet 预测速度快，易于部署的特点，能够很好的在多种设备上完成图像目标检测任务 |
| [Faster-RCNN](https://www.paddlepaddle.org.cn/hubdetail?name=faster_rcnn_coco2017&en_category=ObjectDetection) | 创造性地采用卷积网络自行产生建议框，并且和目标检测网络共享卷积网络，建议框数目减少，质量提高 |
| [YOLOv3](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_coco2017&en_category=ObjectDetection) | 速度和精度均衡的目标检测网络，相比于原作者 darknet 中的 YOLO v3 实现，PaddlePaddle 实现增加了 mixup，label_smooth 等处理，精度 (mAP(0.50: 0.95)) 相比于原作者提高了 4.7 个绝对百分点，在此基础上加入 synchronize batch normalization, 最终精度相比原作者提高 5.9 个绝对百分点。 |
| [人脸检测](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_server&en_category=FaceDetection) | **PyramidBox** **模型是百度自主研发的人脸检测模型**，利用上下文信息解决困难人脸的检测问题，网络表达能力高，鲁棒性强。于18年3月份在 WIDER Face 数据集上取得第一名 |
| [超轻量人脸检测](https://www.paddlepaddle.org.cn/hubdetail?name=ultra_light_fast_generic_face_detector_1mb_640&en_category=FaceDetection) | Ultra-Light-Fast-Generic-Face-Detector-1MB是针对边缘计算设备或低算力设备(如用ARM推理)设计的实时超轻量级通用人脸检测模型，可以在低算力设备中如用ARM进行实时的通用场景的人脸检测推理。该PaddleHub Module的预训练数据集为WIDER FACE数据集，可支持预测，在预测时会将图片输入缩放为640 * 480。 |
| [口罩人脸检测与识别](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_server_mask&en_category=FaceDetection) | 基于PyramidBox而研发的轻量级模型，对于光照、口罩遮挡、表情变化、尺度变化等常见问题具有很强的鲁棒性。基于WIDER FACE数据集和百度自采人脸数据集进行训练，支持预测，可用于检测人脸是否佩戴口罩。 |


#### 图像分割

图像语义分割顾名思义是将图像像素按照表达的语义含义的不同进行分组/分割，图像语义是指对图像内容的理解，例如，能够描绘出什么物体在哪里做了什么事情等，分割是指对图片中的每个像素点进行标注，标注属于哪一类别。近年来用在无人车驾驶技术中分割街景来避让行人和车辆、医疗影像分析中辅助诊断等。
图像语义分割模型请参考语义分割库[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

| 模型名称                                                     | 模型简介                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [deeplabv3](https://www.paddlepaddle.org.cn/hubdetail?name=deeplabv3p_xception65_humanseg&en_category=ImageSegmentation) |DeepLabv3+ 作者通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层， 其骨干网络使用了Xception模型，提高了语义分割的健壮性和运行速率，在 PASCAL VOC 2012 dataset取得新的state-of-art performance。本Module使用百度自建数据集进行训练，可用于人像分割，支持任意大小的图片输入。|
| [ACE2P](https://www.paddlepaddle.org.cn/hubdetail?name=ace2p&en_category=ImageSegmentation) | 人体解析(Human Parsing)是细粒度的语义分割任务，其旨在识别像素级别的人类图像的组成部分（例如，身体部位和服装）。ACE2P通过融合底层特征，全局上下文信息和边缘细节，端到端地训练学习人体解析任务。该结构针对Intersection over Union指标进行针对性的优化学习，提升准确率。以ACE2P单人人体解析网络为基础的解决方案在CVPR2019第三届LIP挑战赛中赢得了全部三个人体解析任务的第一名。该PaddleHub Module采用ResNet101作为骨干网络，接受输入图片大小为473x473x3。 |
| [Pneumonia_CT_LKM_PP](https://www.paddlepaddle.org.cn/hubdetail?name=Pneumonia_CT_LKM_PP&en_category=ImageSegmentation) | 肺炎CT影像分析模型（Pneumonia-CT-LKM-PP）可以高效地完成对患者CT影像的病灶检测识别、病灶轮廓勾画，通过一定的后处理代码，可以分析输出肺部病灶的数量、体积、病灶占比等全套定量指标。值得强调的是，该系统采用的深度学习算法模型充分训练了所收集到的高分辨率和低分辨率的CT影像数据，能极好地适应不同等级CT影像设备采集的检查数据，有望为医疗资源受限和医疗水平偏低的基层医院提供有效的肺炎辅助诊断工具。 |

#### 关键点检测

人体骨骼关键点检测 (Pose Estimation) 主要检测人体的一些关键点，如关节，五官等，通过关键点描述人体骨骼信息。人体骨骼关键点检测对于描述人体姿态，预测人体行为至关重要。是诸多计算机视觉任务的基础，例如动作分类，异常行为检测，以及自动驾驶等等。

| 模型名称                                                     | 模型简介                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Pose Estimation](https://www.paddlepaddle.org.cn/hubdetail?name=human_pose_estimation_resnet50_mpii&en_category=KeyPointDetection) | 人体骨骼关键点检测(Pose Estimation) 是计算机视觉的基础性算法之一，在诸多计算机视觉任务起到了基础性的作用，如行为识别、人物跟踪、步态识别等相关领域。具体应用主要集中在智能视频监控，病人监护系统，人机交互，虚拟现实，人体动画，智能家居，智能安防，运动员辅助训练等等。 该模型的论文《Simple Baselines for Human Pose Estimation and Tracking》由 MSRA 发表于 ECCV18，使用 MPII 数据集训练完成。 |

#### 图像生成

图像生成是指根据输入向量，生成目标图像。这里的输入向量可以是随机的噪声或用户指定的条件向量。具体的应用场景有：手写体生成、人脸合成、风格迁移、图像修复等。[gan](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/gan) 包含和图像生成相关的多个模型。

| 模型名称                                                     | 模型简介                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [CycleGAN](https://www.paddlepaddle.org.cn/hubdetail?name=cyclegan_cityscapes&en_category=GANs) | 图像翻译，可以通过非成对的图片将某一类图片转换成另外一类图片，可用于风格迁移，支持图片从实景图转换为语义分割结果，也支持从语义分割结果转换为实景图。 |
| [StarGAN](https://www.paddlepaddle.org.cn/hubdetail?name=stargan_celeba&en_category=GANs) | 多领域属性迁移，引入辅助分类帮助单个判别器判断多个属性，可用于人脸属性转换。该 PaddleHub Module 使用 Celeba 数据集训练完成，目前支持 "Black_Hair", "Blond_Hair", "Brown_Hair", "Female", "Male", "Aged" 这六种人脸属性转换。 |
| [AttGAN](https://www.paddlepaddle.org.cn/hubdetail?name=attgan_celeba&en_category=GANs) | 利用分类损失和重构损失来保证改变特定的属性，可用于13种人脸特定属性转换。 |
| [STGAN](https://www.paddlepaddle.org.cn/hubdetail?name=stgan_celeba&en_category=GANs) | 人脸特定属性转换，只输入有变化的标签，引入 GRU 结构，更好的选择变化的属性，支持13种属性转换。 |





#### 文本

PaddleNLP 是基于 PaddlePaddle 深度学习框架开发的自然语言处理 (NLP) 工具，算法，模型和数据的开源项目。百度在 NLP 领域十几年的深厚积淀为 PaddleNLP 提供了强大的核心动力。


#### 中文词法分析与词向量
| 模型名称                                                     | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [LAC 中文词法分析](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis) | Lexical Analysis of Chinese，简称LAC，是百度自主研发中文特色模型词法分析任务，集成了中文分词、词性标注和命名实体识别任务。输入是一个字符串，而输出是句子中的词边界和词性、实体类别。 |
| [word2vec词向量](https://www.paddlepaddle.org.cn/hubdetail?name=word2vec_skipgram&en_category=SemanticModel) | Word2vec是常用的词嵌入（word embedding）模型。该PaddleHub Module基于Skip-gram模型，在海量百度搜索数据集下预训练得到中文单词预训练词嵌入。其支持Fine-tune。Word2vec的预训练数据集的词汇表大小为1700249，word embedding维度为128。 |

#### 情感分析

| 模型名称                                                     | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Senta](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis) | 情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度，能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。该模型基于一个双向LSTM结构，情感类型分为积极、消极。该PaddleHub Module支持预测和Fine-tune。 |
| [emotion_detection](https://www.paddlepaddle.org.cn/hubdetail?name=emotion_detection_textcnn&en_category=SentimentAnalysis) | 对话情绪识别（Emotion Detection，简称EmoTect）专注于识别智能对话场景中用户的情绪，针对智能对话场景中的用户文本，自动判断该文本的情绪类别并给出相应的置信度，情绪类型分为积极、消极、中性。该模型基于TextCNN（多卷积核CNN模型），能够更好地捕捉句子局部相关性。该PaddleHub Module预训练数据集为百度自建数据集，支持预测和Fine-tune。 |

#### 文本相似度计算

[SimNet](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleNLP/similarity_net) (Similarity Net) 是一个计算短文本相似度的框架，主要包括 BOW、CNN、RNN、MMDNN 等核心网络结构形式。SimNet 框架在百度各产品上广泛应用，提供语义相似度计算训练和预测框架，适用于信息检索、新闻推荐、智能客服等多个应用场景，帮助企业解决语义匹配问题。

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [simnet_bow](https://www.paddlepaddle.org.cn/hubdetail?name=simnet_bow&en_category=SemanticModel) | SimNet是一个计算短文本相似度的模型，可以根据用户输入的两个文本，计算出相似度得分。该PaddleHub Module基于百度海量搜索数据进行训练，支持命令行和Python接口进行预测 |

#### 文本审核

文本审核也是NLP方向的一个常用任务，可以广泛应用在各种信息分发平台、论坛、讨论区的文本审核中。

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [porn_detection_gru](https://www.paddlepaddle.org.cn/hubdetail?name=porn_detection_gru&en_category=TextCensorship) | 色情检测模型可自动判别文本是否涉黄并给出相应的置信度，对文本中的色情描述、低俗交友、污秽文爱进行识别。porn_detection_gru采用GRU网络结构并按字粒度进行切词。该模型最大句子长度为256字，仅支持预测。 |

#### 语义表示

通过在大规模语料上训练得到的通用的语义表示模型，可以助益其他自然语言处理任务，是通用预训练 + 特定任务精调范式的体现。PaddleLARK 集成了 ELMO，BERT，ERNIE 1.0，ERNIE 2.0，XLNet 等热门中英文预训练模型。

| 模型                                                         | 简介                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ERNIE](https://www.paddlepaddle.org.cn/hubdetail?name=ERNIE&en_category=SemanticModel) | ERNIE通过建模海量数据中的词、实体及实体关系，学习真实世界的语义知识。相较于BERT学习原始语言信号，ERNIE直接对先验语义知识单元进行建模，增强了模型语义表示能力，以Transformer为网络基本组件，以Masked Bi-Language Model和Next Sentence Prediction为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的NLP任务，在多个任务上取得了SOTA的结果。其可用于文本分类、序列标注、阅读理解等任务。预训练数据集为百科类、资讯类、论坛对话类数据等中文语料。该PaddleHub Module可支持Fine-tune。 |
| [BERT](https://www.paddlepaddle.org.cn/hubdetail?name=bert_multi_uncased_L-12_H-768_A-12&en_category=SemanticModel) | 一个迁移能力很强的通用语义表示模型， 以 Transformer 为网络基本组件，以双向 Masked Language Model和 Next Sentence Prediction 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务，在多个任务上取得了 SOTA 的结果。 |
| [RoBERTa](https://www.paddlepaddle.org.cn/hubdetail?name=rbtl3&en_category=SemanticModel) | RoBERTa (a Robustly Optimized BERT Pretraining Approach) 是BERT通用语义表示模型的一个优化版，它在BERT模型的基础上提出了Dynamic Masking方法、去除了Next Sentence Prediction目标，同时在更多的数据上采用更大的batch size训练更长的时间，在多个任务中做到了SOTA。rbtl3以roberta_wwm_ext_chinese_L-24_H-1024_A-16模型参数初始化前三层Transformer以及词向量层并在此基础上继续训练了1M步，在仅损失少量效果的情况下大幅减少参数量，得到推断速度的进一步提升。当该PaddleHub Module用于Fine-tune时，其输入是单文本（如Fine-tune的任务为情感分类等）或文本对（如Fine-tune任务为文本语义相似度匹配等），可用于文本分类、序列标注、阅读理解等任务。 |
| [chinese-bert](https://www.paddlepaddle.org.cn/hubdetail?name=chinese-bert-wwm&en_category=SemanticModel) | chinese_bert_wwm是支持中文的BERT模型，它采用全词遮罩（Whole Word Masking）技术，考虑到了中文分词问题。预训练数据集为中文维基百科。该PaddleHub Module只支持Fine-tune。当该PaddleHub Module用于Fine-tune时，其输入是单文本（如Fine-tune的任务为情感分类等）或文本对（如Fine-tune任务为文本语义相似度匹配等），可用于文本分类、序列标注、阅读理解等任务。 |



## 视频

视频数据包含语音、图像等多种信息，因此理解视频任务不仅需要处理语音和图像，还需要提取视频帧时间序列中的上下文信息。视频分类模型提供了提取全局时序特征的方法，主要方式有卷积神经网络 (C3D, I3D, C2D等)，神经网络和传统图像算法结合 (VLAD 等)，循环神经网络等建模方法。视频动作定位模型需要同时识别视频动作的类别和起止时间点，通常采用类似于图像目标检测中的算法在时间维度上进行建模。


| 模型名称                                                     | 模型简介                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| videotag_tsn_lstm                                            | videotag_tsn_lstm是一个基于千万短视频预训练的视频分类模型，可直接预测短视频的中文标签。模型分为视频特征抽取和序列建模两个阶段，前者使用TSN网络提取视频特征，后者基于前者输出使用AttentionLSTM网络进行序列建模实现分类。模型基于百度实际短视频场景中的大规模数据训练得到，在实际业务中取得89.9%的Top-1精度，同时具有良好的泛化能力，适用于多种短视频中文标签分类场景。该PaddleHub Module可支持预测。 |
| [TSN](https://www.paddlepaddle.org.cn/hubdetail?name=tsn_kinetics400&en_category=VideoClassification) | TSN（Temporal Segment Network）是视频分类领域经典的基于2D-CNN的解决方案。该方法主要解决视频的长时间行为判断问题，通过稀疏采样视频帧的方式代替稠密采样，既能捕获视频全局信息，也能去除冗余，降低计算量。最终将每帧特征平均融合后得到视频的整体特征，并用于分类。TSN的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。该PaddleHub Module可支持预测。 |
| [Non-Local](https://www.paddlepaddle.org.cn/hubdetail?name=tsn_kinetics400&en_category=VideoClassification) | Non-local Neural Networks是由Xiaolong Wang等研究者在2017年提出的模型，主要特点是通过引入Non-local操作来描述距离较远的像素点之间的关联关系。其借助于传统计算机视觉中的non-local mean的思想，并将该思想扩展到神经网络中，通过定义输出位置和所有输入位置之间的关联函数，建立全局关联特性。Non-local模型的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。该PaddleHub Module可支持预测。 |
| [StNet](https://www.paddlepaddle.org.cn/hubdetail?name=stnet_kinetics400&en_category=VideoClassification) | StNet模型框架为ActivityNet Kinetics Challenge 2018中夺冠的基础网络框架，是基于ResNet50实现的。该模型提出super-image的概念，在super-image上进行2D卷积，建模视频中局部时空相关性。另外通过temporal modeling block建模视频的全局时空依赖，最后用一个temporal Xception block对抽取的特征序列进行长时序建模。StNet的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。该PaddleHub Module可支持预测。 |
| [TSM](https://www.paddlepaddle.org.cn/hubdetail?name=tsm_kinetics400&en_category=VideoClassification) | TSM（Temporal Shift Module）是由MIT和IBM Watson AI Lab的JiLin，ChuangGan和SongHan等人提出的通过时间位移来提高网络视频理解能力的模块。TSM的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。该PaddleHub Module可支持预测。 |
