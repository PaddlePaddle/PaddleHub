## **For better user experience, refer to the official web documentation -> [Image Classification](https://www.paddlepaddle.org.cn/hublist)**

### Image Classification

Image classification distinguishes between different categories of images based on the semantic information. This is an important basic issue in computer vision and is the basis for object detection, image segmentation, object tracking, behavior analysis, face recognition and other high-level vision tasks. It is widely applied in many fields. For example, face recognition and intelligent video analysis in the security field, traffic scene recognition in the transportation field, content-based image retrieval and automatic classification of photo albums in the Internet field, and image recognition in the medicine field.

**Note:** If you are an experienced developer, feel free to use as required. If you are a newcomer, choose Resnet50 first at the server side and MobileNetV3 for mobile.

- Recommended Models

|                      | **Model Name**                                               | **Model Features**                                           |
| -------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| Image Classification | [Dish Identification](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_dishes&en_category=ImageClassification) | Private dataset training. It supports the category identification of 8416 dishes. It is suitable for further fine-tuning in the dish orientation. |
| Image Classification | [Animal Identification](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification) | Private dataset training. It supports the category identification of 7978 kinds of animals. It is suitable for further fine-tuning in the animal orientation. |
| Image Classification | [Wildlife product identification](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_wildanimals&en_category=ImageClassification) | Supports identification of the ten tags 'Ivory product', 'Ivory', 'Elephant', 'Tiger Skin', 'Tiger', 'Tiger Tusk/Claw/Bone', 'Pangolin squama', 'Pangolin', 'Pangolin Paw',  and 'Other'. |

- More Models

| **Model Name**                                               | **Model Introduction**                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [AlexNet](https://www.paddlepaddle.org.cn/hubdetail?name=alexnet_imagenet&en_category=ImageClassification) | It is the first successful application of ReLU, Dropout and LRN in CNN, with using GPU for acceleration. |
| [VGG19](https://www.paddlepaddle.org.cn/hubdetail?name=vgg19_imagenet&en_category=ImageClassification) | Use 3x3 small convolutional cores based on AlexNet to increase network depth with good generalization capabilities |
| [GoogLeNet](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/image_classification) | Increased network depth and width without increasing computational load for superior performance |
| [ResNet50](https://www.paddlepaddle.org.cn/hubdetail?name=resnet_v2_50_imagenet&en_category=ImageClassification) | Residual Network. Introduce a new residual structure that solves the problem of decreasing accuracy as the network deepens. |
| [Inceptionv4](https://www.paddlepaddle.org.cn/hubdetail?name=inception_v4_imagenet&en_category=ImageClassification) | Combine the Inception module with the Residual Connection greatly. This accelerates training and performance increases through ResNet's architecture. |
| [MobileNetV2](https://www.paddlepaddle.org.cn/hubdetail?name=mobilenet_v2_imagenet&en_category=ImageClassification) | Fine-tuning of the MobileNet structure. Perform the skip learning connections directly on the bottleneck layer of thinner. There is no ReLu nonlinear processing of the bottleneck layer, with a good results. |
| [se_resnext50](https://www.paddlepaddle.org.cn/hubdetail?name=se_resnext50_32x4d_imagenet&en_category=ImageClassification) | Add the SE (Sequeeze-and-Excitation) module to ResNeXt to improve recognition accuracy. Achieve the first place in the ILSVRC 2017 classification program. |
| [ShuffleNetV2](https://www.paddlepaddle.org.cn/hubdetail?name=shufflenet_v2_imagenet&en_category=ImageClassification) | ECCV2018, lightweight CNN network. It offers a good balance between speed and accuracy. It is more accurate than ShuffleNet and MobileNetv2 at the same level of complexity. It is more suitable for mobile and unmanned vehicle fields. |
| [efficientNetb7](https://www.paddlepaddle.org.cn/hubdetail?name=efficientnetb7_imagenet&en_category=ImageClassification) | Simultaneous resolution for models through the scaling of number of channels and depth. SOTA's accuracy can be achieved with very few parameters. |
| [xception71](https://www.paddlepaddle.org.cn/hubdetail?name=xception71_imagenet&en_category=ImageClassification) | Improvements to inceptiv-v3. Replace regular convolution with deeply separable convolution. It reduces the number of parameters and increases the accuracy. |
| [dpn107](https://www.paddlepaddle.org.cn/hubdetail?name=dpn107_imagenet&en_category=ImageClassification) | A fusion of densenet and resnext features.                   |
| [DarkNet53](https://www.paddlepaddle.org.cn/hubdetail?name=darknet53_imagenet&en_category=ImageClassification) | The detection framework yolov3 uses backbone. It has good performance for both classification and detection tasks. |
| [DenseNet161](https://www.paddlepaddle.org.cn/hubdetail?name=densenet161_imagenet&en_category=ImageClassification) | A densely connected network structure is proposed, which is more conducive to the flow of information. |
| [ResNeXt152_vd](https://www.paddlepaddle.org.cn/hubdetail?name=resnext152_64x4d_imagenet&en_category=ImageClassification) | The concept of cardinatity is proposed as an additional measure of model complexity, effectively improving model accuracy. |
