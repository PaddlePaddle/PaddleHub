English | [简体中文](README_ch.md)

# CONTENTS
|[Image](#Image) (212)|[Text](#Text) (130)|[Audio](#Audio) (15)|[Video](#Video) (8)|[Industrial Application](#Industrial-Application) (1)|
|--|--|--|--|--|
|[Image Classification](#Image-Classification) (108)|[Text Generation](#Text-Generation) (17)| [Voice Cloning](#Voice-Cloning) (2)|[Video Classification](#Video-Classification) (5)| [Meter Detection](#Meter-Detection) (1)|
|[Image Generation](#Image-Generation) (26)|[Word Embedding](#Word-Embedding) (62)|[Text to Speech](#Text-to-Speech) (5)|[Video Editing](#Video-Editing) (1)|-|
|[Keypoint Detection](#Keypoint-Detection) (5)|[Machine Translation](#Machine-Translation) (2)|[Automatic Speech Recognition](#Automatic-Speech-Recognition) (5)|[Multiple Object tracking](#Multiple-Object-tracking) (2)|-|
|[Semantic Segmentation](#Semantic-Segmentation) (25)|[Language Model](#Language-Model) (30)|[Audio Classification](#Audio-Classification) (3)| -|-|
|[Face Detection](#Face-Detection) (7)|[Sentiment Analysis](#Sentiment-Analysis) (7)|-|-|-|
|[Text Recognition](#Text-Recognition) (17)|[Syntactic Analysis](#Syntactic-Analysis) (1)|-|-|-|
|[Image Editing](#Image-Editing) (8)|[Simultaneous Translation](#Simultaneous-Translation) (5)|-|-|-|
|[Instance Segmentation](#Instance-Segmentation) (1)|[Lexical Analysis](#Lexical-Analysis) (2)|-|-|-|
|[Object Detection](#Object-Detection) (13)|[Punctuation Restoration](#Punctuation-Restoration) (1)|-|-|-|
|[Depth Estimation](#Depth-Estimation) (2)|[Text Review](#Text-Review) (3)|-|-|-|

## Image
  - ### Image Classification

<details><summary>expand</summary><div>

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[DriverStatusRecognition](image/classification/DriverStatusRecognition)|MobileNetV3_small_ssld|分心司机检测数据集||
|[mobilenet_v2_animals](image/classification/mobilenet_v2_animals)|MobileNet_v2|百度自建动物数据集||
|[repvgg_a1_imagenet](image/classification/repvgg_a1_imagenet)|RepVGG|ImageNet-2012||
|[repvgg_a0_imagenet](image/classification/repvgg_a0_imagenet)|RepVGG|ImageNet-2012||
|[resnext152_32x4d_imagenet](image/classification/resnext152_32x4d_imagenet)|ResNeXt|ImageNet-2012||
|[resnet_v2_152_imagenet](image/classification/resnet_v2_152_imagenet)|ResNet V2|ImageNet-2012||
|[resnet50_vd_animals](image/classification/resnet50_vd_animals)|ResNet50_vd|百度自建动物数据集||
|[food_classification](image/classification/food_classification)|ResNet50_vd_ssld|美食数据集||
|[mobilenet_v3_large_imagenet_ssld](image/classification/mobilenet_v3_large_imagenet_ssld)|Mobilenet_v3_large|ImageNet-2012||
|[resnext152_vd_32x4d_imagenet](image/classification/resnext152_vd_32x4d_imagenet)||||
|[ghostnet_x1_3_imagenet_ssld](image/classification/ghostnet_x1_3_imagenet_ssld)|GhostNet|ImageNet-2012||
|[rexnet_1_5_imagenet](image/classification/rexnet_1_5_imagenet)|ReXNet|ImageNet-2012||
|[resnext50_64x4d_imagenet](image/classification/resnext50_64x4d_imagenet)|ResNeXt|ImageNet-2012||
|[resnext101_64x4d_imagenet](image/classification/resnext101_64x4d_imagenet)|ResNeXt|ImageNet-2012||
|[efficientnetb0_imagenet](image/classification/efficientnetb0_imagenet)|EfficientNet|ImageNet-2012||
|[efficientnetb1_imagenet](image/classification/efficientnetb1_imagenet)|EfficientNet|ImageNet-2012||
|[mobilenet_v2_imagenet_ssld](image/classification/mobilenet_v2_imagenet_ssld)|Mobilenet_v2|ImageNet-2012||
|[resnet50_vd_dishes](image/classification/resnet50_vd_dishes)|ResNet50_vd|百度自建菜品数据集||
|[pnasnet_imagenet](image/classification/pnasnet_imagenet)|PNASNet|ImageNet-2012||
|[rexnet_2_0_imagenet](image/classification/rexnet_2_0_imagenet)|ReXNet|ImageNet-2012||
|[SnakeIdentification](image/classification/SnakeIdentification)|ResNet50_vd_ssld|蛇种数据集||
|[hrnet40_imagenet](image/classification/hrnet40_imagenet)|HRNet|ImageNet-2012||
|[resnet_v2_34_imagenet](image/classification/resnet_v2_34_imagenet)|ResNet V2|ImageNet-2012||
|[mobilenet_v2_dishes](image/classification/mobilenet_v2_dishes)|MobileNet_v2|百度自建菜品数据集||
|[resnext101_vd_32x4d_imagenet](image/classification/resnext101_vd_32x4d_imagenet)|ResNeXt|ImageNet-2012||
|[repvgg_b2g4_imagenet](image/classification/repvgg_b2g4_imagenet)|RepVGG|ImageNet-2012||
|[fix_resnext101_32x48d_wsl_imagenet](image/classification/fix_resnext101_32x48d_wsl_imagenet)|ResNeXt|ImageNet-2012||
|[vgg13_imagenet](image/classification/vgg13_imagenet)|VGG|ImageNet-2012||
|[se_resnext101_32x4d_imagenet](image/classification/se_resnext101_32x4d_imagenet)|SE_ResNeXt|ImageNet-2012||
|[hrnet30_imagenet](image/classification/hrnet30_imagenet)|HRNet|ImageNet-2012||
|[ghostnet_x1_3_imagenet](image/classification/ghostnet_x1_3_imagenet)|GhostNet|ImageNet-2012||
|[dpn107_imagenet](image/classification/dpn107_imagenet)|DPN|ImageNet-2012||
|[densenet161_imagenet](image/classification/densenet161_imagenet)|DenseNet|ImageNet-2012||
|[vgg19_imagenet](image/classification/vgg19_imagenet)|vgg19_imagenet|ImageNet-2012||
|[mobilenet_v2_imagenet](image/classification/mobilenet_v2_imagenet)|Mobilenet_v2|ImageNet-2012||
|[resnet50_vd_10w](image/classification/resnet50_vd_10w)|ResNet_vd|百度自建数据集||
|[resnet_v2_101_imagenet](image/classification/resnet_v2_101_imagenet)|ResNet V2 101|ImageNet-2012||
|[darknet53_imagenet](image/classification/darknet53_imagenet)|DarkNet|ImageNet-2012||
|[se_resnext50_32x4d_imagenet](image/classification/se_resnext50_32x4d_imagenet)|SE_ResNeXt|ImageNet-2012||
|[se_hrnet64_imagenet_ssld](image/classification/se_hrnet64_imagenet_ssld)|HRNet|ImageNet-2012||
|[resnext101_32x16d_wsl](image/classification/resnext101_32x16d_wsl)|ResNeXt_wsl|ImageNet-2012||
|[hrnet18_imagenet](image/classification/hrnet18_imagenet)|HRNet|ImageNet-2012||
|[spinalnet_res101_gemstone](image/classification/spinalnet_res101_gemstone)|resnet101|gemstone||
|[densenet264_imagenet](image/classification/densenet264_imagenet)|DenseNet|ImageNet-2012||
|[resnext50_vd_32x4d_imagenet](image/classification/resnext50_vd_32x4d_imagenet)|ResNeXt_vd|ImageNet-2012||
|[SpinalNet_Gemstones](image/classification/SpinalNet_Gemstones)||||
|[spinalnet_vgg16_gemstone](image/classification/spinalnet_vgg16_gemstone)|vgg16|gemstone||
|[xception71_imagenet](image/classification/xception71_imagenet)|Xception|ImageNet-2012||
|[repvgg_b2_imagenet](image/classification/repvgg_b2_imagenet)|RepVGG|ImageNet-2012||
|[dpn68_imagenet](image/classification/dpn68_imagenet)|DPN|ImageNet-2012||
|[alexnet_imagenet](image/classification/alexnet_imagenet)|AlexNet|ImageNet-2012||
|[rexnet_1_3_imagenet](image/classification/rexnet_1_3_imagenet)|ReXNet|ImageNet-2012||
|[hrnet64_imagenet](image/classification/hrnet64_imagenet)|HRNet|ImageNet-2012||
|[efficientnetb7_imagenet](image/classification/efficientnetb7_imagenet)|EfficientNet|ImageNet-2012||
|[efficientnetb0_small_imagenet](image/classification/efficientnetb0_small_imagenet)|EfficientNet|ImageNet-2012||
|[efficientnetb6_imagenet](image/classification/efficientnetb6_imagenet)|EfficientNet|ImageNet-2012||
|[hrnet48_imagenet](image/classification/hrnet48_imagenet)|HRNet|ImageNet-2012||
|[rexnet_3_0_imagenet](image/classification/rexnet_3_0_imagenet)|ReXNet|ImageNet-2012||
|[shufflenet_v2_imagenet](image/classification/shufflenet_v2_imagenet)|ShuffleNet V2|ImageNet-2012||
|[ghostnet_x0_5_imagenet](image/classification/ghostnet_x0_5_imagenet)|GhostNet|ImageNet-2012||
|[inception_v4_imagenet](image/classification/inception_v4_imagenet)|Inception_V4|ImageNet-2012||
|[resnext101_vd_64x4d_imagenet](image/classification/resnext101_vd_64x4d_imagenet)|ResNeXt_vd|ImageNet-2012||
|[densenet201_imagenet](image/classification/densenet201_imagenet)|DenseNet|ImageNet-2012||
|[vgg16_imagenet](image/classification/vgg16_imagenet)|VGG|ImageNet-2012||
|[mobilenet_v3_small_imagenet_ssld](image/classification/mobilenet_v3_small_imagenet_ssld)|Mobilenet_v3_Small|ImageNet-2012||
|[hrnet18_imagenet_ssld](image/classification/hrnet18_imagenet_ssld)|HRNet|ImageNet-2012||
|[resnext152_64x4d_imagenet](image/classification/resnext152_64x4d_imagenet)|ResNeXt|ImageNet-2012||
|[efficientnetb3_imagenet](image/classification/efficientnetb3_imagenet)|EfficientNet|ImageNet-2012||
|[efficientnetb2_imagenet](image/classification/efficientnetb2_imagenet)|EfficientNet|ImageNet-2012||
|[repvgg_b1g4_imagenet](image/classification/repvgg_b1g4_imagenet)|RepVGG|ImageNet-2012||
|[resnext101_32x4d_imagenet](image/classification/resnext101_32x4d_imagenet)|ResNeXt|ImageNet-2012||
|[resnext50_32x4d_imagenet](image/classification/resnext50_32x4d_imagenet)|ResNeXt|ImageNet-2012||
|[repvgg_a2_imagenet](image/classification/repvgg_a2_imagenet)|RepVGG|ImageNet-2012||
|[resnext152_vd_64x4d_imagenet](image/classification/resnext152_vd_64x4d_imagenet)|ResNeXt_vd|ImageNet-2012||
|[xception41_imagenet](image/classification/xception41_imagenet)|Xception|ImageNet-2012||
|[googlenet_imagenet](image/classification/googlenet_imagenet)|GoogleNet|ImageNet-2012||
|[resnet50_vd_imagenet_ssld](image/classification/resnet50_vd_imagenet_ssld)|ResNet_vd|ImageNet-2012||
|[repvgg_b1_imagenet](image/classification/repvgg_b1_imagenet)|RepVGG|ImageNet-2012||
|[repvgg_b0_imagenet](image/classification/repvgg_b0_imagenet)|RepVGG|ImageNet-2012||
|[resnet_v2_50_imagenet](image/classification/resnet_v2_50_imagenet)|ResNet V2|ImageNet-2012||
|[rexnet_1_0_imagenet](image/classification/rexnet_1_0_imagenet)|ReXNet|ImageNet-2012||
|[resnet_v2_18_imagenet](image/classification/resnet_v2_18_imagenet)|ResNet V2|ImageNet-2012||
|[resnext101_32x8d_wsl](image/classification/resnext101_32x8d_wsl)|ResNeXt_wsl|ImageNet-2012||
|[efficientnetb4_imagenet](image/classification/efficientnetb4_imagenet)|EfficientNet|ImageNet-2012||
|[efficientnetb5_imagenet](image/classification/efficientnetb5_imagenet)|EfficientNet|ImageNet-2012||
|[repvgg_b1g2_imagenet](image/classification/repvgg_b1g2_imagenet)|RepVGG|ImageNet-2012||
|[resnext101_32x48d_wsl](image/classification/resnext101_32x48d_wsl)|ResNeXt_wsl|ImageNet-2012||
|[resnet50_vd_wildanimals](image/classification/resnet50_vd_wildanimals)|ResNet_vd|IFAW 自建野生动物数据集||
|[nasnet_imagenet](image/classification/nasnet_imagenet)|NASNet|ImageNet-2012||
|[se_resnet18_vd_imagenet](image/classification/se_resnet18_vd_imagenet)||||
|[spinalnet_res50_gemstone](image/classification/spinalnet_res50_gemstone)|resnet50|gemstone||
|[resnext50_vd_64x4d_imagenet](image/classification/resnext50_vd_64x4d_imagenet)|ResNeXt_vd|ImageNet-2012||
|[resnext101_32x32d_wsl](image/classification/resnext101_32x32d_wsl)|ResNeXt_wsl|ImageNet-2012||
|[dpn131_imagenet](image/classification/dpn131_imagenet)|DPN|ImageNet-2012||
|[xception65_imagenet](image/classification/xception65_imagenet)|Xception|ImageNet-2012||
|[repvgg_b3g4_imagenet](image/classification/repvgg_b3g4_imagenet)|RepVGG|ImageNet-2012||
|[marine_biometrics](image/classification/marine_biometrics)|ResNet50_vd_ssld|Fish4Knowledge||
|[res2net101_vd_26w_4s_imagenet](image/classification/res2net101_vd_26w_4s_imagenet)|Res2Net|ImageNet-2012||
|[dpn98_imagenet](image/classification/dpn98_imagenet)|DPN|ImageNet-2012||
|[resnet18_vd_imagenet](image/classification/resnet18_vd_imagenet)|ResNet_vd|ImageNet-2012||
|[densenet121_imagenet](image/classification/densenet121_imagenet)|DenseNet|ImageNet-2012||
|[vgg11_imagenet](image/classification/vgg11_imagenet)|VGG|ImageNet-2012||
|[hrnet44_imagenet](image/classification/hrnet44_imagenet)|HRNet|ImageNet-2012||
|[densenet169_imagenet](image/classification/densenet169_imagenet)|DenseNet|ImageNet-2012||
|[hrnet32_imagenet](image/classification/hrnet32_imagenet)|HRNet|ImageNet-2012||
|[dpn92_imagenet](image/classification/dpn92_imagenet)|DPN|ImageNet-2012||
|[ghostnet_x1_0_imagenet](image/classification/ghostnet_x1_0_imagenet)|GhostNet|ImageNet-2012||
|[hrnet48_imagenet_ssld](image/classification/hrnet48_imagenet_ssld)|HRNet|ImageNet-2012||

</div></details>


  - ### Image Generation

|module|Network|Dataset|Introduction| Huggingface Spaces Demo|
|--|--|--|--|--|
|[pixel2style2pixel](image/Image_gan/gan/pixel2style2pixel/)|Pixel2Style2Pixel|-|人脸转正|
|[stgan_bald](image/Image_gan/gan/stgan_bald/)|STGAN|CelebA|秃头生成器| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/stgan_bald) |
|[styleganv2_editing](image/Image_gan/gan/styleganv2_editing)|StyleGAN V2|-|人脸编辑|
|[wav2lip](image/Image_gan/gan/wav2lip)|wav2lip|LRS2|唇形生成| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/wav2lip) |
|[attgan_celeba](image/Image_gan/attgan_celeba/)|AttGAN|Celeba|人脸编辑|
|[cyclegan_cityscapes](image/Image_gan/cyclegan_cityscapes)|CycleGAN|Cityscapes|实景图和语义分割结果互相转换|
|[stargan_celeba](image/Image_gan/stargan_celeba)|StarGAN|Celeba|人脸编辑|
|[stgan_celeba](image/Image_gan/stgan_celeba/)|STGAN|Celeba|人脸编辑|
|[ID_Photo_GEN](image/Image_gan/style_transfer/ID_Photo_GEN)|HRNet_W18|-|证件照生成|
|[Photo2Cartoon](image/Image_gan/style_transfer/Photo2Cartoon)|U-GAT-IT|cartoon_data|人脸卡通化|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/photo2cartoon) |
|[U2Net_Portrait](image/Image_gan/style_transfer/U2Net_Portrait)|U^2Net|-|人脸素描化|
|[UGATIT_100w](image/Image_gan/style_transfer/UGATIT_100w)|U-GAT-IT|selfie2anime|人脸动漫化|
|[UGATIT_83w](image/Image_gan/style_transfer/UGATIT_83w)|U-GAT-IT|selfie2anime|人脸动漫化|
|[UGATIT_92w](image/Image_gan/style_transfer/UGATIT_92w)| U-GAT-IT|selfie2anime|人脸动漫化|
|[animegan_v1_hayao_60](image/Image_gan/style_transfer/animegan_v1_hayao_60)|AnimeGAN|The Wind Rises|图像风格迁移-宫崎骏| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v1_hayao_60) |
|[animegan_v2_hayao_64](image/Image_gan/style_transfer/animegan_v2_hayao_64)|AnimeGAN|The Wind Rises|图像风格迁移-宫崎骏| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_hayao_64) |
|[animegan_v2_hayao_99](image/Image_gan/style_transfer/animegan_v2_hayao_99)|AnimeGAN|The Wind Rises|图像风格迁移-宫崎骏| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_hayao_99) |
|[animegan_v2_paprika_54](image/Image_gan/style_transfer/animegan_v2_paprika_54)|AnimeGAN|Paprika|图像风格迁移-今敏| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_54) |
|[animegan_v2_paprika_74](image/Image_gan/style_transfer/animegan_v2_paprika_74)|AnimeGAN|Paprika|图像风格迁移-今敏| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_74) |
|[animegan_v2_paprika_97](image/Image_gan/style_transfer/animegan_v2_paprika_97)|AnimeGAN|Paprika|图像风格迁移-今敏| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_97) |
|[animegan_v2_paprika_98](image/Image_gan/style_transfer/animegan_v2_paprika_98)|AnimeGAN|Paprika|图像风格迁移-今敏| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_98) |
|[animegan_v2_shinkai_33](image/Image_gan/style_transfer/animegan_v2_shinkai_33)|AnimeGAN|Your Name, Weathering with you|图像风格迁移-新海诚| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_shinkai_33) |
|[animegan_v2_shinkai_53](image/Image_gan/style_transfer/animegan_v2_shinkai_53)|AnimeGAN|Your Name, Weathering with you|图像风格迁移-新海诚| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_shinkai_53) |
|[msgnet](image/Image_gan/style_transfer/msgnet)|msgnet|COCO2014| |[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/msgnet) |
|[stylepro_artistic](image/Image_gan/style_transfer/stylepro_artistic)|StyleProNet|MS-COCO + WikiArt|艺术风格迁移| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/stylepro_artistic) |
|stylegan_ffhq|StyleGAN|FFHQ|图像风格迁移|

  - ### Keypoint Detection

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[face_landmark_localization](image/keypoint_detection/face_landmark_localization)|Face_Landmark|AFW/AFLW|人脸关键点检测|
|[hand_pose_localization](image/keypoint_detection/hand_pose_localization)|-|MPII, NZSL|手部关键点检测|
|[openpose_body_estimation](image/keypoint_detection/openpose_body_estimation)|two-branch multi-stage CNN|MPII, COCO 2016|肢体关键点检测|
|[human_pose_estimation_resnet50_mpii](image/keypoint_detection/human_pose_estimation_resnet50_mpii)|Pose_Resnet50|MPII|人体骨骼关键点检测
|[openpose_hands_estimation](image/keypoint_detection/openpose_hands_estimation)|-|MPII, NZSL|手部关键点检测|

  - ### Semantic Segmentation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[deeplabv3p_xception65_humanseg](image/semantic_segmentation/deeplabv3p_xception65_humanseg)|deeplabv3p|百度自建数据集|人像分割|
|[humanseg_server](image/semantic_segmentation/humanseg_server)|deeplabv3p|百度自建数据集|人像分割|
|[humanseg_mobile](image/semantic_segmentation/humanseg_mobile)|hrnet|百度自建数据集|人像分割-移动端前置摄像头|
|[humanseg_lite](image/semantic_segmentation/umanseg_lite)|shufflenet|百度自建数据集|轻量级人像分割-移动端实时|
|[ExtremeC3_Portrait_Segmentation](image/semantic_segmentation/ExtremeC3_Portrait_Segmentation)|ExtremeC3|EG1800, Baidu fashion dataset|轻量化人像分割|
|[SINet_Portrait_Segmentation](image/semantic_segmentation/SINet_Portrait_Segmentation)|SINet|EG1800, Baidu fashion dataset|轻量化人像分割|
|[FCN_HRNet_W18_Face_Seg](image/semantic_segmentation/FCN_HRNet_W18_Face_Seg)|FCN_HRNet_W18|-|人像分割|
|[ace2p](image/semantic_segmentation/ace2p)|ACE2P|LIP|人体解析|
|[Pneumonia_CT_LKM_PP](image/semantic_segmentation/Pneumonia_CT_LKM_PP)|U-NET+|连心医疗授权脱敏数据集|肺炎CT影像分析|
|[Pneumonia_CT_LKM_PP_lung](image/semantic_segmentation/Pneumonia_CT_LKM_PP_lung)|U-NET+|连心医疗授权脱敏数据集|肺炎CT影像分析|
|[ocrnet_hrnetw18_voc](image/semantic_segmentation/ocrnet_hrnetw18_voc)|ocrnet, hrnet|PascalVoc2012|
|[U2Net](image/semantic_segmentation/U2Net)|U^2Net|-|图像前景背景分割|
|[U2Netp](image/semantic_segmentation/U2Netp)|U^2Net|-|图像前景背景分割|
|[Extract_Line_Draft](image/semantic_segmentation/Extract_Line_Draft)|UNet|Pixiv|线稿提取|
|[unet_cityscapes](image/semantic_segmentation/unet_cityscapes)|UNet|cityscapes|
|[ocrnet_hrnetw18_cityscapes](image/semantic_segmentation/ocrnet_hrnetw18_cityscapes)|ocrnet_hrnetw18|cityscapes|
|[hardnet_cityscapes](image/semantic_segmentation/hardnet_cityscapes)|hardnet|cityscapes|
|[fcn_hrnetw48_voc](image/semantic_segmentation/fcn_hrnetw48_voc)|fcn_hrnetw48|PascalVoc2012|
|[fcn_hrnetw48_cityscapes](image/semantic_segmentation/fcn_hrnetw48_cityscapes)|fcn_hrnetw48|cityscapes|
|[fcn_hrnetw18_voc](image/semantic_segmentation/fcn_hrnetw18_voc)|fcn_hrnetw18|PascalVoc2012|
|[fcn_hrnetw18_cityscapes](image/semantic_segmentation/fcn_hrnetw18_cityscapes)|fcn_hrnetw18|cityscapes|
|[fastscnn_cityscapes](image/semantic_segmentation/fastscnn_cityscapes)|fastscnn|cityscapes|
|[deeplabv3p_resnet50_voc](image/semantic_segmentation/deeplabv3p_resnet50_voc)|deeplabv3p, resnet50|PascalVoc2012|
|[deeplabv3p_resnet50_cityscapes](image/semantic_segmentation/deeplabv3p_resnet50_cityscapes)|deeplabv3p, resnet50|cityscapes|
|[bisenetv2_cityscapes](image/semantic_segmentation/bisenetv2_cityscapes)|bisenetv2|cityscapes|



  - ### Face Detection

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[pyramidbox_lite_mobile](image/face_detection/pyramidbox_lite_mobile)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|轻量级人脸检测-移动端|
|[pyramidbox_lite_mobile_mask](image/face_detection/pyramidbox_lite_mobile_mask)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|轻量级人脸口罩检测-移动端|
|[pyramidbox_lite_server_mask](image/face_detection/pyramidbox_lite_server_mask)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|轻量级人脸口罩检测|
|[ultra_light_fast_generic_face_detector_1mb_640](image/face_detection/ultra_light_fast_generic_face_detector_1mb_640)|Ultra-Light-Fast-Generic-Face-Detector-1MB|WIDER FACE数据集|轻量级通用人脸检测-低算力设备|
|[ultra_light_fast_generic_face_detector_1mb_320](image/face_detection/ultra_light_fast_generic_face_detector_1mb_320)|Ultra-Light-Fast-Generic-Face-Detector-1MB|WIDER FACE数据集|轻量级通用人脸检测-低算力设备|
|[pyramidbox_lite_server](image/face_detection/pyramidbox_lite_server)|PyramidBox|WIDER FACE数据集 + 百度自采人脸数据集|轻量级人脸检测|
|[pyramidbox_face_detection](image/face_detection/pyramidbox_face_detection)|PyramidBox|WIDER FACE数据集|人脸检测|

  - ### Text Recognition

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[chinese_ocr_db_crnn_mobile](image/text_recognition/chinese_ocr_db_crnn_mobile)|Differentiable Binarization+RCNN|icdar2015数据集|中文文字识别|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/chinese_ocr_db_crnn_mobile) |[chinese_text_detection_db_mobile](image/text_recognition/chinese_text_detection_db_mobile)|Differentiable Binarization|icdar2015数据集|中文文本检测|
|[chinese_text_detection_db_server](image/text_recognition/chinese_text_detection_db_server)|Differentiable Binarization|icdar2015数据集|中文文本检测|
|[chinese_ocr_db_crnn_server](image/text_recognition/chinese_ocr_db_crnn_server)|Differentiable Binarization+RCNN|icdar2015数据集|中文文字识别|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/chinese_ocr_db_crnn_server) |
|[Vehicle_License_Plate_Recognition](image/text_recognition/Vehicle_License_Plate_Recognition)|-|CCPD|车牌识别|
|[chinese_cht_ocr_db_crnn_mobile](image/text_recognition/chinese_cht_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|繁体中文文字识别|
|[japan_ocr_db_crnn_mobile](image/text_recognition/japan_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|日文文字识别|
|[korean_ocr_db_crnn_mobile](image/text_recognition/korean_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|韩文文字识别|
|[german_ocr_db_crnn_mobile](image/text_recognition/german_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|德文文字识别|
|[french_ocr_db_crnn_mobile](image/text_recognition/french_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|法文文字识别|
|[latin_ocr_db_crnn_mobile](image/text_recognition/latin_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|拉丁文文字识别|
|[cyrillic_ocr_db_crnn_mobile](image/text_recognition/cyrillic_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|斯拉夫文文字识别|
|[multi_languages_ocr_db_crnn](image/text_recognition/multi_languages_ocr_db_crnn)|Differentiable Binarization+RCNN|icdar2015数据集|多语言文字识别|
|[kannada_ocr_db_crnn_mobile](image/text_recognition/kannada_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|卡纳达文文字识别|
|[arabic_ocr_db_crnn_mobile](image/text_recognition/arabic_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|阿拉伯文文字识别|
|[telugu_ocr_db_crnn_mobile](image/text_recognition/telugu_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|泰卢固文文字识别|
|[devanagari_ocr_db_crnn_mobile](image/text_recognition/devanagari_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|梵文文字识别|
|[tamil_ocr_db_crnn_mobile](image/text_recognition/tamil_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015数据集|泰米尔文文字识别|


  - ### Image Editing

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[realsr](image/Image_editing/super_resolution/realsr)|LP-KPN|RealSR dataset|图像/视频超分-4倍|
|[deoldify](image/Image_editing/colorization/deoldify)|GAN|ILSVRC 2012|黑白照片/视频着色|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/deoldify) |
|[photo_restoration](image/Image_editing/colorization/photo_restoration)|基于deoldify和realsr模型|-|老照片修复|
|[user_guided_colorization](image/Image_editing/colorization/user_guided_colorization)|siggraph|ILSVRC 2012|图像着色|
|[falsr_c](image/Image_editing/super_resolution/falsr_c)|falsr_c| DIV2k|轻量化超分-2倍|
|[dcscn](image/Image_editing/super_resolution/dcscn)|dcscn| DIV2k|轻量化超分-2倍|
|[falsr_a](image/Image_editing/super_resolution/falsr_a)|falsr_a| DIV2k|轻量化超分-2倍|
|[falsr_b](image/Image_editing/super_resolution/falsr_b)|falsr_b|DIV2k|轻量化超分-2倍|

  - ### Instance Segmentation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[solov2](image/instance_segmentation/solov2)|-|COCO2014|实例分割|

  - ### Object Detection

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[faster_rcnn_resnet50_coco2017](image/object_detection/faster_rcnn_resnet50_coco2017)|faster_rcnn|COCO2017||
|[ssd_vgg16_512_coco2017](image/object_detection/ssd_vgg16_512_coco2017)|SSD|COCO2017||
|[faster_rcnn_resnet50_fpn_venus](image/object_detection/faster_rcnn_resnet50_fpn_venus)|faster_rcnn|百度自建数据集|大规模通用目标检测|
|[ssd_vgg16_300_coco2017](image/object_detection/ssd_vgg16_300_coco2017)||||
|[yolov3_resnet34_coco2017](image/object_detection/yolov3_resnet34_coco2017)|YOLOv3|COCO2017||
|[yolov3_darknet53_pedestrian](image/object_detection/yolov3_darknet53_pedestrian)|YOLOv3|百度自建大规模行人数据集|行人检测|
|[yolov3_mobilenet_v1_coco2017](image/object_detection/yolov3_mobilenet_v1_coco2017)|YOLOv3|COCO2017||
|[ssd_mobilenet_v1_pascal](image/object_detection/ssd_mobilenet_v1_pascal)|SSD|PASCAL VOC||
|[faster_rcnn_resnet50_fpn_coco2017](image/object_detection/faster_rcnn_resnet50_fpn_coco2017)|faster_rcnn|COCO2017||
|[yolov3_darknet53_coco2017](image/object_detection/yolov3_darknet53_coco2017)|YOLOv3|COCO2017||
|[yolov3_darknet53_vehicles](image/object_detection/yolov3_darknet53_vehicles)|YOLOv3|百度自建大规模车辆数据集|车辆检测|
|[yolov3_darknet53_venus](image/object_detection/yolov3_darknet53_venus)|YOLOv3|百度自建数据集|大规模通用检测|
|[yolov3_resnet50_vd_coco2017](image/object_detection/yolov3_resnet50_vd_coco2017)|YOLOv3|COCO2017||

  - ### Depth Estimation

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[MiDaS_Large](image/depth_estimation/MiDaS_Large)|-|3D Movies, WSVD, ReDWeb, MegaDepth|| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/MiDaS_Large) |
|[MiDaS_Small](image/depth_estimation/MiDaS_Small)|-|3D Movies, WSVD, ReDWeb, MegaDepth, etc.|| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/MiDaS_Small) |

## Text
  - ### Text Generation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[ernie_gen](text/text_generation/ernie_gen)|ERNIE-GEN|-|面向生成任务的预训练-微调框架|
|[ernie_gen_poetry](text/text_generation/ernie_gen_poetry)|ERNIE-GEN|开源诗歌数据集|诗歌生成|
|[ernie_gen_couplet](text/text_generation/ernie_gen_couplet)|ERNIE-GEN|开源对联数据集|对联生成|
|[ernie_gen_lover_words](text/text_generation/ernie_gen_lover_words)|ERNIE-GEN|网络情诗、情话数据|情话生成|
|[ernie_tiny_couplet](text/text_generation/ernie_tiny_couplet)|Eernie_tiny|开源对联数据集|对联生成|
|[ernie_gen_acrostic_poetry](text/text_generation/ernie_gen_acrostic_poetry)|ERNIE-GEN|开源诗歌数据集|藏头诗生成|
|[Rumor_prediction](text/text_generation/Rumor_prediction)|-|新浪微博中文谣言数据|谣言预测|
|[plato-mini](text/text_generation/plato-mini)|Unified Transformer|十亿级别的中文对话数据|中文对话|
|[plato2_en_large](text/text_generation/plato2_en_large)|plato2|开放域多轮数据集|超大规模生成式对话|
|[plato2_en_base](text/text_generation/plato2_en_base)|plato2|开放域多轮数据集|超大规模生成式对话|
|[CPM_LM](text/text_generation/CPM_LM)|GPT-2|自建数据集|中文文本生成|
|[unified_transformer-12L-cn](text/text_generation/unified_transformer-12L-cn)|Unified Transformer|千万级别中文会话数据|人机多轮对话|
|[unified_transformer-12L-cn-luge](text/text_generation/unified_transformer-12L-cn-luge)|Unified Transformer|千言对话数据集|人机多轮对话|
|[reading_pictures_writing_poems](text/text_generation/reading_pictures_writing_poems)|多网络级联|-|看图写诗|
|[GPT2_CPM_LM](text/text_generation/GPT2_CPM_LM)|||问答类文本生成|
|[GPT2_Base_CN](text/text_generation/GPT2_Base_CN)|||问答类文本生成|

  - ### Word Embedding

<details><summary>expand</summary><div>

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[w2v_weibo_target_word-bigram_dim300](text/embedding/w2v_weibo_target_word-bigram_dim300)|w2v|weibo||
|[w2v_baidu_encyclopedia_target_word-ngram_1-2_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-ngram_1-2_dim300)|w2v|baidu_encyclopedia||
|[w2v_literature_target_word-word_dim300](text/embedding/w2v_literature_target_word-word_dim300)|w2v|literature||
|[word2vec_skipgram](text/embedding/word2vec_skipgram)|skip-gram|百度自建数据集||
|[w2v_sogou_target_word-char_dim300](text/embedding/w2v_sogou_target_word-char_dim300)|w2v|sogou||
|[w2v_weibo_target_bigram-char_dim300](text/embedding/w2v_weibo_target_bigram-char_dim300)|w2v|weibo||
|[w2v_zhihu_target_word-bigram_dim300](text/embedding/w2v_zhihu_target_word-bigram_dim300)|w2v|zhihu||
|[w2v_financial_target_word-word_dim300](text/embedding/w2v_financial_target_word-word_dim300)|w2v|financial||
|[w2v_wiki_target_word-word_dim300](text/embedding/w2v_wiki_target_word-word_dim300)|w2v|wiki||
|[w2v_baidu_encyclopedia_context_word-word_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-word_dim300)|w2v|baidu_encyclopedia||
|[w2v_weibo_target_word-word_dim300](text/embedding/w2v_weibo_target_word-word_dim300)|w2v|weibo||
|[w2v_zhihu_target_bigram-char_dim300](text/embedding/w2v_zhihu_target_bigram-char_dim300)|w2v|zhihu||
|[w2v_zhihu_target_word-word_dim300](text/embedding/w2v_zhihu_target_word-word_dim300)|w2v|zhihu||
|[w2v_people_daily_target_word-char_dim300](text/embedding/w2v_people_daily_target_word-char_dim300)|w2v|people_daily||
|[w2v_sikuquanshu_target_word-word_dim300](text/embedding/w2v_sikuquanshu_target_word-word_dim300)|w2v|sikuquanshu||
|[glove_twitter_target_word-word_dim200_en](text/embedding/glove_twitter_target_word-word_dim200_en)|fasttext|twitter||
|[fasttext_crawl_target_word-word_dim300_en](text/embedding/fasttext_crawl_target_word-word_dim300_en)|fasttext|crawl||
|[w2v_wiki_target_word-bigram_dim300](text/embedding/w2v_wiki_target_word-bigram_dim300)|w2v|wiki||
|[w2v_baidu_encyclopedia_context_word-character_char1-1_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-character_char1-1_dim300)|w2v|baidu_encyclopedia||
|[glove_wiki2014-gigaword_target_word-word_dim300_en](text/embedding/glove_wiki2014-gigaword_target_word-word_dim300_en)|glove|wiki2014-gigaword||
|[glove_wiki2014-gigaword_target_word-word_dim50_en](text/embedding/glove_wiki2014-gigaword_target_word-word_dim50_en)|glove|wiki2014-gigaword||
|[w2v_baidu_encyclopedia_context_word-ngram_2-2_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-ngram_2-2_dim300)|w2v|baidu_encyclopedia||
|[w2v_wiki_target_bigram-char_dim300](text/embedding/w2v_wiki_target_bigram-char_dim300)|w2v|wiki||
|[w2v_baidu_encyclopedia_target_word-character_char1-1_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-character_char1-1_dim300)|w2v|baidu_encyclopedia||
|[w2v_financial_target_bigram-char_dim300](text/embedding/w2v_financial_target_bigram-char_dim300)|w2v|financial||
|[glove_wiki2014-gigaword_target_word-word_dim200_en](text/embedding/glove_wiki2014-gigaword_target_word-word_dim200_en)|glove|wiki2014-gigaword||
|[w2v_financial_target_word-bigram_dim300](text/embedding/w2v_financial_target_word-bigram_dim300)|w2v|financial||
|[w2v_mixed-large_target_word-char_dim300](text/embedding/w2v_mixed-large_target_word-char_dim300)|w2v|mixed||
|[w2v_baidu_encyclopedia_target_word-wordPosition_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-wordPosition_dim300)|w2v|baidu_encyclopedia||
|[w2v_baidu_encyclopedia_context_word-ngram_1-3_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-ngram_1-3_dim300)|w2v|baidu_encyclopedia||
|[w2v_baidu_encyclopedia_target_word-wordLR_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-wordLR_dim300)|w2v|baidu_encyclopedia||
|[w2v_sogou_target_bigram-char_dim300](text/embedding/w2v_sogou_target_bigram-char_dim300)|w2v|sogou||
|[w2v_weibo_target_word-char_dim300](text/embedding/w2v_weibo_target_word-char_dim300)|w2v|weibo||
|[w2v_people_daily_target_word-word_dim300](text/embedding/w2v_people_daily_target_word-word_dim300)|w2v|people_daily||
|[w2v_zhihu_target_word-char_dim300](text/embedding/w2v_zhihu_target_word-char_dim300)|w2v|zhihu||
|[w2v_wiki_target_word-char_dim300](text/embedding/w2v_wiki_target_word-char_dim300)|w2v|wiki||
|[w2v_sogou_target_word-bigram_dim300](text/embedding/w2v_sogou_target_word-bigram_dim300)|w2v|sogou||
|[w2v_financial_target_word-char_dim300](text/embedding/w2v_financial_target_word-char_dim300)|w2v|financial||
|[w2v_baidu_encyclopedia_target_word-ngram_1-3_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-ngram_1-3_dim300)|w2v|baidu_encyclopedia||
|[glove_wiki2014-gigaword_target_word-word_dim100_en](text/embedding/glove_wiki2014-gigaword_target_word-word_dim100_en)|glove|wiki2014-gigaword||
|[w2v_baidu_encyclopedia_target_word-character_char1-4_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-character_char1-4_dim300)|w2v|baidu_encyclopedia||
|[w2v_sogou_target_word-word_dim300](text/embedding/w2v_sogou_target_word-word_dim300)|w2v|sogou||
|[w2v_literature_target_word-char_dim300](text/embedding/w2v_literature_target_word-char_dim300)|w2v|literature||
|[w2v_baidu_encyclopedia_target_bigram-char_dim300](text/embedding/w2v_baidu_encyclopedia_target_bigram-char_dim300)|w2v|baidu_encyclopedia||
|[w2v_baidu_encyclopedia_target_word-word_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-word_dim300)|w2v|baidu_encyclopedia||
|[glove_twitter_target_word-word_dim100_en](text/embedding/glove_twitter_target_word-word_dim100_en)|glove|crawl||
|[w2v_baidu_encyclopedia_target_word-ngram_2-2_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-ngram_2-2_dim300)|w2v|baidu_encyclopedia||
|[w2v_baidu_encyclopedia_context_word-character_char1-4_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-character_char1-4_dim300)|w2v|baidu_encyclopedia||
|[w2v_literature_target_bigram-char_dim300](text/embedding/w2v_literature_target_bigram-char_dim300)|w2v|literature||
|[fasttext_wiki-news_target_word-word_dim300_en](text/embedding/fasttext_wiki-news_target_word-word_dim300_en)|fasttext|wiki-news||
|[w2v_people_daily_target_word-bigram_dim300](text/embedding/w2v_people_daily_target_word-bigram_dim300)|w2v|people_daily||
|[w2v_mixed-large_target_word-word_dim300](text/embedding/w2v_mixed-large_target_word-word_dim300)|w2v|mixed||
|[w2v_people_daily_target_bigram-char_dim300](text/embedding/w2v_people_daily_target_bigram-char_dim300)|w2v|people_daily||
|[w2v_literature_target_word-bigram_dim300](text/embedding/w2v_literature_target_word-bigram_dim300)|w2v|literature||
|[glove_twitter_target_word-word_dim25_en](text/embedding/glove_twitter_target_word-word_dim25_en)|glove|twitter||
|[w2v_baidu_encyclopedia_context_word-ngram_1-2_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-ngram_1-2_dim300)|w2v|baidu_encyclopedia||
|[w2v_sikuquanshu_target_word-bigram_dim300](text/embedding/w2v_sikuquanshu_target_word-bigram_dim300)|w2v|sikuquanshu||
|[w2v_baidu_encyclopedia_context_word-character_char1-2_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-character_char1-2_dim300)|w2v|baidu_encyclopedia||
|[glove_twitter_target_word-word_dim50_en](text/embedding/glove_twitter_target_word-word_dim50_en)|glove|twitter||
|[w2v_baidu_encyclopedia_context_word-wordLR_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-wordLR_dim300)|w2v|baidu_encyclopedia||
|[w2v_baidu_encyclopedia_target_word-character_char1-2_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-character_char1-2_dim300)|w2v|baidu_encyclopedia||
|[w2v_baidu_encyclopedia_context_word-wordPosition_dim300](text/embedding/w2v_baidu_encyclopedia_context_word-wordPosition_dim300)|w2v|baidu_encyclopedia||

</div></details>

  - ### Machine Translation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[transformer_zh-en](text/machine_translation/transformer/zh-en)|Transformer|CWMT2021|中文译英文|
|[transformer_en-de](text/machine_translation/transformer/en-de)|Transformer|WMT14 EN-DE|英文译德文|

  - ### Language Model

<details><summary>expand</summary><div>

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[chinese_electra_small](text/language_model/chinese_electra_small)||||
|[chinese_electra_base](text/language_model/chinese_electra_base)||||
|[roberta-wwm-ext-large](text/language_model/roberta-wwm-ext-large)|roberta-wwm-ext-large|百度自建数据集||
|[chinese-bert-wwm-ext](text/language_model/chinese_bert_wwm_ext)|chinese-bert-wwm-ext|百度自建数据集||
|[lda_webpage](text/language_model/lda_webpage)|LDA|百度自建网页领域数据集||
|[lda_novel](text/language_model/lda_novel)||||
|[bert-base-multilingual-uncased](text/language_model/bert-base-multilingual-uncased)||||
|[rbt3](text/language_model/rbt3)||||
|[ernie_v2_eng_base](text/language_model/ernie_v2_eng_base)|ernie_v2_eng_base|百度自建数据集||
|[bert-base-multilingual-cased](text/language_model/bert-base-multilingual-cased)||||
|[rbtl3](text/language_model/rbtl3)||||
|[chinese-bert-wwm](text/language_model/chinese_bert_wwm)|chinese-bert-wwm|百度自建数据集||
|[bert-large-uncased](text/language_model/bert-large-uncased)||||
|[slda_novel](text/language_model/slda_novel)||||
|[slda_news](text/language_model/slda_news)||||
|[electra_small](text/language_model/electra_small)||||
|[slda_webpage](text/language_model/slda_webpage)||||
|[bert-base-cased](text/language_model/bert-base-cased)||||
|[slda_weibo](text/language_model/slda_weibo)||||
|[roberta-wwm-ext](text/language_model/roberta-wwm-ext)|roberta-wwm-ext|百度自建数据集||
|[bert-base-uncased](text/language_model/bert-base-uncased)||||
|[electra_large](text/language_model/electra_large)||||
|[ernie](text/language_model/ernie)|ernie-1.0|百度自建数据集||
|[simnet_bow](text/language_model/simnet_bow)|BOW|百度自建数据集||
|[ernie_tiny](text/language_model/ernie_tiny)|ernie_tiny|百度自建数据集||
|[bert-base-chinese](text/language_model/bert-base-chinese)|bert-base-chinese|百度自建数据集||
|[lda_news](text/language_model/lda_news)|LDA|百度自建新闻领域数据集||
|[electra_base](text/language_model/electra_base)||||
|[ernie_v2_eng_large](text/language_model/ernie_v2_eng_large)|ernie_v2_eng_large|百度自建数据集||
|[bert-large-cased](text/language_model/bert-large-cased)||||

</div></details>


  - ### Sentiment Analysis

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[ernie_skep_sentiment_analysis](text/sentiment_analysis/ernie_skep_sentiment_analysis)|SKEP|百度自建数据集|句子级情感分析|
|[emotion_detection_textcnn](text/sentiment_analysis/emotion_detection_textcnn)|TextCNN|百度自建数据集|对话情绪识别|
|[senta_bilstm](text/sentiment_analysis/senta_bilstm)|BiLSTM|百度自建数据集|中文情感倾向分析| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/senta_bilstm) 
|[senta_bow](text/sentiment_analysis/senta_bow)|BOW|百度自建数据集|中文情感倾向分析|
|[senta_gru](text/sentiment_analysis/senta_gru)|GRU|百度自建数据集|中文情感倾向分析|
|[senta_lstm](text/sentiment_analysis/senta_lstm)|LSTM|百度自建数据集|中文情感倾向分析|
|[senta_cnn](text/sentiment_analysis/senta_cnn)|CNN|百度自建数据集|中文情感倾向分析|

  - ### Syntactic Analysis

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[DDParser](text/syntactic_analysis/DDParser)|Deep Biaffine Attention|搜索query、网页文本、语音输入等数据|句法分析|

  - ### Simultaneous Translation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[transformer_nist_wait_1](text/simultaneous_translation/stacl/transformer_nist_wait_1)|transformer|NIST 2008-中英翻译数据集|中译英-wait-1策略|
|[transformer_nist_wait_3](text/simultaneous_translation/stacl/transformer_nist_wait_3)|transformer|NIST 2008-中英翻译数据集|中译英-wait-3策略|
|[transformer_nist_wait_5](text/simultaneous_translation/stacl/transformer_nist_wait_5)|transformer|NIST 2008-中英翻译数据集|中译英-wait-5策略|
|[transformer_nist_wait_7](text/simultaneous_translation/stacl/transformer_nist_wait_7)|transformer|NIST 2008-中英翻译数据集|中译英-wait-7策略|
|[transformer_nist_wait_all](text/simultaneous_translation/stacl/transformer_nist_wait_all)|transformer|NIST 2008-中英翻译数据集|中译英-waitk=-1策略|


  - ### Lexical Analysis

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[jieba_paddle](text/lexical_analysis/jieba_paddle)|BiGRU+CRF|百度自建数据集|jieba使用Paddle搭建的切词网络（双向GRU）。同时支持jieba的传统切词方法，如精确模式、全模式、搜索引擎模式等切词模式。|
|[lac](text/lexical_analysis/lac)|BiGRU+CRF|百度自建数据集|百度自研联合的词法分析模型，能整体性地完成中文分词、词性标注、专名识别任务。在百度自建数据集上评测，LAC效果：Precision=88.0%，Recall=88.7%，F1-Score=88.4%。|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/lac) 

  - ### Punctuation Restoration

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[auto_punc](text/punctuation_restoration/auto_punc)|Ernie-1.0|WuDaoCorpora 2.0|自动添加7种标点符号|

  - ### Text Review

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[porn_detection_cnn](text/text_review/porn_detection_cnn)|CNN|百度自建数据集|色情检测，自动判别文本是否涉黄并给出相应的置信度，对文本中的色情描述、低俗交友、污秽文案进行识别|
|[porn_detection_gru](text/text_review/porn_detection_gru)|GRU|百度自建数据集|色情检测，自动判别文本是否涉黄并给出相应的置信度，对文本中的色情描述、低俗交友、污秽文案进行识别|
|[porn_detection_lstm](text/text_review/porn_detection_lstm)|LSTM|百度自建数据集|色情检测，自动判别文本是否涉黄并给出相应的置信度，对文本中的色情描述、低俗交友、污秽文案进行识别|

## Audio

  - ### Voice cloning

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[ge2e_fastspeech2_pwgan](audio/voice_cloning/ge2e_fastspeech2_pwgan)|FastSpeech2|AISHELL-3|中文语音克隆|
|[lstm_tacotron2](audio/voice_cloning/lstm_tacotron2)|LSTM、Tacotron2、WaveFlow|AISHELL-3|中文语音克隆|

  - ### Text to Speech

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[transformer_tts_ljspeech](audio/tts/transformer_tts_ljspeech)|Transformer|LJSpeech-1.1|英文语音合成|
|[fastspeech_ljspeech](audio/tts/fastspeech_ljspeech)|FastSpeech|LJSpeech-1.1|英文语音合成|
|[fastspeech2_baker](audio/tts/fastspeech2_baker)|FastSpeech2|Chinese Standard Mandarin Speech Copus|中文语音合成|
|[fastspeech2_ljspeech](audio/tts/fastspeech2_ljspeech)|FastSpeech2|LJSpeech-1.1|英文语音合成|
|[deepvoice3_ljspeech](audio/tts/deepvoice3_ljspeech)|DeepVoice3|LJSpeech-1.1|英文语音合成|

  - ### Automatic Speech Recognition

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[deepspeech2_aishell](audio/asr/deepspeech2_aishell)|DeepSpeech2|AISHELL-1|中文语音识别|
|[deepspeech2_librispeech](audio/asr/deepspeech2_librispeech)|DeepSpeech2|LibriSpeech|英文语音识别|
|[u2_conformer_aishell](audio/asr/u2_conformer_aishell)|Conformer|AISHELL-1|中文语音识别|
|[u2_conformer_wenetspeech](audio/asr/u2_conformer_wenetspeech)|Conformer|WenetSpeech|中文语音识别|
|[u2_conformer_librispeech](audio/asr/u2_conformer_librispeech)|Conformer|LibriSpeech|英文语音识别|


  - ### Audio Classification

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[panns_cnn6](audio/audio_classification/PANNs/cnn6)|PANNs|Google Audioset|主要包含4个卷积层和2个全连接层，模型参数为4.5M。经过预训练后，可以用于提取音频的embbedding，维度是512|
|[panns_cnn14](audio/audio_classification/PANNs/cnn14)|PANNs|Google Audioset|主要包含12个卷积层和2个全连接层，模型参数为79.6M。经过预训练后，可以用于提取音频的embbedding，维度是2048|
|[panns_cnn10](audio/audio_classification/PANNs/cnn10)|PANNs|Google Audioset|主要包含8个卷积层和2个全连接层，模型参数为4.9M。经过预训练后，可以用于提取音频的embbedding，维度是512|

## Video
  - ### Video Classification

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[videotag_tsn_lstm](video/classification/videotag_tsn_lstm)|TSN + AttentionLSTM|百度自建数据集|大规模短视频分类打标签|
|[tsn_kinetics400](video/classification/tsn_kinetics400)|TSN|Kinetics-400|视频分类|
|[tsm_kinetics400](video/classification/tsm_kinetics400)|TSM|Kinetics-400|视频分类|
|[stnet_kinetics400](video/classification/stnet_kinetics400)|StNet|Kinetics-400|视频分类|
|[nonlocal_kinetics400](video/classification/nonlocal_kinetics400)|Non-local|Kinetics-400|视频分类|


  - ### Video Editing

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[SkyAR](video/Video_editing/SkyAR)|UNet|UNet|视频换天|

  - ### Multiple Object tracking

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[fairmot_dla34](video/multiple_object_tracking/fairmot_dla34)|CenterNet|Caltech Pedestrian+CityPersons+CUHK-SYSU+PRW+ETHZ+MOT17|实时多目标跟踪|
|[jde_darknet53](video/multiple_object_tracking/jde_darknet53)|YOLOv3|Caltech Pedestrian+CityPersons+CUHK-SYSU+PRW+ETHZ+MOT17|多目标跟踪-兼顾精度和速度|

## Industrial Application

  - ### Meter Detection

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[WatermeterSegmentation](image/semantic_segmentation/WatermeterSegmentation)|DeepLabV3|水表的数字表盘分割数据集|水表的数字表盘分割|
