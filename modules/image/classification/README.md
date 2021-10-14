
## **更好用户体验，建议参考WEB端官方文档 -> [【图像分类】](https://www.paddlepaddle.org.cn/hublist)**


### 图像分类
图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉中重要的基础问题，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层视觉任务的基础，在许多领域都有着广泛的应用。如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

**注：** **如果你是资深开发者，那可以随意按需使用**，**假如你是新手，服务器端优先选择Resnet50，移动端优先选择MobileNetV3**

- 精选模型推荐

|            | **模型名称**                                                 | **模型特色**                                       |
| ---------- | :----------------------------------------------------------- | ---------------------------------------------------------- |
| 图像分类 | [菜品识别](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_dishes&en_category=ImageClassification) | 私有数据集训练，支持8416种菜品的分类识别，适合进一步菜品方向微调 |
| 图像分类 | [动物识别](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification) | 私有数据集训练，支持7978种动物的分类识别，适合进一步动物方向微调 |
| 图像分类 | [野生动物制品识别](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_wildanimals&en_category=ImageClassification) | 支持'象牙制品', '象牙', '大象', '虎皮', '老虎', '虎牙/虎爪/虎骨', '穿山甲甲片', '穿山甲', '穿山甲爪子', '其他' 这十个标签的识别。 |


- 更多模型

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
