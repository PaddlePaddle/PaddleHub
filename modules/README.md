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
|[DriverStatusRecognition](image/classification/DriverStatusRecognition)|MobileNetV3_small_ssld|Drivers||
|[mobilenet_v2_animals](image/classification/mobilenet_v2_animals)|MobileNet_v2|Animals||
|[repvgg_a1_imagenet](image/classification/repvgg_a1_imagenet)|RepVGG|ImageNet-2012||
|[repvgg_a0_imagenet](image/classification/repvgg_a0_imagenet)|RepVGG|ImageNet-2012||
|[resnext152_32x4d_imagenet](image/classification/resnext152_32x4d_imagenet)|ResNeXt|ImageNet-2012||
|[resnet_v2_152_imagenet](image/classification/resnet_v2_152_imagenet)|ResNet V2|ImageNet-2012||
|[resnet50_vd_animals](image/classification/resnet50_vd_animals)|ResNet50_vd|Animals||
|[food_classification](image/classification/food_classification)|ResNet50_vd_ssld|dishes||
|[mobilenet_v3_large_imagenet_ssld](image/classification/mobilenet_v3_large_imagenet_ssld)|Mobilenet_v3_large|ImageNet-2012||
|[resnext152_vd_32x4d_imagenet](image/classification/resnext152_vd_32x4d_imagenet)||||
|[ghostnet_x1_3_imagenet_ssld](image/classification/ghostnet_x1_3_imagenet_ssld)|GhostNet|ImageNet-2012||
|[rexnet_1_5_imagenet](image/classification/rexnet_1_5_imagenet)|ReXNet|ImageNet-2012||
|[resnext50_64x4d_imagenet](image/classification/resnext50_64x4d_imagenet)|ResNeXt|ImageNet-2012||
|[resnext101_64x4d_imagenet](image/classification/resnext101_64x4d_imagenet)|ResNeXt|ImageNet-2012||
|[efficientnetb0_imagenet](image/classification/efficientnetb0_imagenet)|EfficientNet|ImageNet-2012||
|[efficientnetb1_imagenet](image/classification/efficientnetb1_imagenet)|EfficientNet|ImageNet-2012||
|[mobilenet_v2_imagenet_ssld](image/classification/mobilenet_v2_imagenet_ssld)|Mobilenet_v2|ImageNet-2012||
|[resnet50_vd_dishes](image/classification/resnet50_vd_dishes)|ResNet50_vd|dishes||
|[pnasnet_imagenet](image/classification/pnasnet_imagenet)|PNASNet|ImageNet-2012||
|[rexnet_2_0_imagenet](image/classification/rexnet_2_0_imagenet)|ReXNet|ImageNet-2012||
|[SnakeIdentification](image/classification/SnakeIdentification)|ResNet50_vd_ssld|snakes||
|[hrnet40_imagenet](image/classification/hrnet40_imagenet)|HRNet|ImageNet-2012||
|[resnet_v2_34_imagenet](image/classification/resnet_v2_34_imagenet)|ResNet V2|ImageNet-2012||
|[mobilenet_v2_dishes](image/classification/mobilenet_v2_dishes)|MobileNet_v2|dishes||
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
|[resnet50_vd_10w](image/classification/resnet50_vd_10w)|ResNet_vd|private||
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
|[resnet50_vd_wildanimals](image/classification/resnet50_vd_wildanimals)|ResNet_vd|IFAW wild animals||
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
|[pixel2style2pixel](image/Image_gan/gan/pixel2style2pixel/)|Pixel2Style2Pixel|-|human face|
|[stgan_bald](image/Image_gan/gan/stgan_bald/)|STGAN|CelebA|stgan_bald| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/stgan_bald) |
|[styleganv2_editing](image/Image_gan/gan/styleganv2_editing)|StyleGAN V2|-|human face editing|
|[wav2lip](image/Image_gan/gan/wav2lip)|wav2lip|LRS2|wav2lip| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/wav2lip) |
|[attgan_celeba](image/Image_gan/attgan_celeba/)|AttGAN|Celeba|human face editing|
|[cyclegan_cityscapes](image/Image_gan/cyclegan_cityscapes)|CycleGAN|Cityscapes|cyclegan_cityscapes|
|[stargan_celeba](image/Image_gan/stargan_celeba)|StarGAN|Celeba|human face editing|
|[stgan_celeba](image/Image_gan/stgan_celeba/)|STGAN|Celeba|human face editing|
|[ID_Photo_GEN](image/Image_gan/style_transfer/ID_Photo_GEN)|HRNet_W18|-|ID_Photo_GEN|
|[Photo2Cartoon](image/Image_gan/style_transfer/Photo2Cartoon)|U-GAT-IT|cartoon_data|cartoon|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/photo2cartoon) |
|[U2Net_Portrait](image/Image_gan/style_transfer/U2Net_Portrait)|U^2Net|-|Portrait|
|[UGATIT_100w](image/Image_gan/style_transfer/UGATIT_100w)|U-GAT-IT|selfie2anime|selfie2anime|
|[UGATIT_83w](image/Image_gan/style_transfer/UGATIT_83w)|U-GAT-IT|selfie2anime|selfie2anime|
|[UGATIT_92w](image/Image_gan/style_transfer/UGATIT_92w)| U-GAT-IT|selfie2anime|selfie2anime|
|[animegan_v1_hayao_60](image/Image_gan/style_transfer/animegan_v1_hayao_60)|AnimeGAN|The Wind Rises|animegan_v1_hayao| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v1_hayao_60) |
|[animegan_v2_hayao_64](image/Image_gan/style_transfer/animegan_v2_hayao_64)|AnimeGAN|The Wind Rises|animegan_v1_hayao| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_hayao_64) |
|[animegan_v2_hayao_99](image/Image_gan/style_transfer/animegan_v2_hayao_99)|AnimeGAN|The Wind Rises|animegan_v1_hayao| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_hayao_99) |
|[animegan_v2_paprika_54](image/Image_gan/style_transfer/animegan_v2_paprika_54)|AnimeGAN|Paprika|animegan_v2_paprika| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_54) |
|[animegan_v2_paprika_74](image/Image_gan/style_transfer/animegan_v2_paprika_74)|AnimeGAN|Paprika|animegan_v2_paprika| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_74) |
|[animegan_v2_paprika_97](image/Image_gan/style_transfer/animegan_v2_paprika_97)|AnimeGAN|Paprika|animegan_v2_paprika| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_97) |
|[animegan_v2_paprika_98](image/Image_gan/style_transfer/animegan_v2_paprika_98)|AnimeGAN|Paprika|animegan_v2_paprika| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_paprika_98) |
|[animegan_v2_shinkai_33](image/Image_gan/style_transfer/animegan_v2_shinkai_33)|AnimeGAN|Your Name, Weathering with you|animegan_v2_shinkai| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_shinkai_33) |
|[animegan_v2_shinkai_53](image/Image_gan/style_transfer/animegan_v2_shinkai_53)|AnimeGAN|Your Name, Weathering with you|animegan_v2_shinkai| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/animegan_v2_shinkai_53) |
|[msgnet](image/Image_gan/style_transfer/msgnet)|msgnet|COCO2014| |[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/msgnet) |
|[stylepro_artistic](image/Image_gan/style_transfer/stylepro_artistic)|StyleProNet|MS-COCO + WikiArt|stylepro_artistic| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/stylepro_artistic) |
|stylegan_ffhq|StyleGAN|FFHQ|stylepro_artistic|

  - ### Keypoint Detection

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[face_landmark_localization](image/keypoint_detection/face_landmark_localization)|Face_Landmark|AFW/AFLW|Face_Landmark|
|[hand_pose_localization](image/keypoint_detection/hand_pose_localization)|-|MPII, NZSL|hand_pose_localization|
|[openpose_body_estimation](image/keypoint_detection/openpose_body_estimation)|two-branch multi-stage CNN|MPII, COCO 2016|openpose_body_estimation|
|[human_pose_estimation_resnet50_mpii](image/keypoint_detection/human_pose_estimation_resnet50_mpii)|Pose_Resnet50|MPII|human_pose_estimation
|[openpose_hands_estimation](image/keypoint_detection/openpose_hands_estimation)|-|MPII, NZSL|openpose_hands_estimation|

  - ### Semantic Segmentation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[deeplabv3p_xception65_humanseg](image/semantic_segmentation/deeplabv3p_xception65_humanseg)|deeplabv3p|-|humanseg|
|[humanseg_server](image/semantic_segmentation/humanseg_server)|deeplabv3p|-|humanseg|
|[humanseg_mobile](image/semantic_segmentation/humanseg_mobile)|hrnet|-|humanseg|
|[humanseg_lite](image/semantic_segmentation/umanseg_lite)|shufflenet|-|humanseg|
|[ExtremeC3_Portrait_Segmentation](image/semantic_segmentation/ExtremeC3_Portrait_Segmentation)|ExtremeC3|EG1800, Baidu fashion dataset|humanseg|
|[SINet_Portrait_Segmentation](image/semantic_segmentation/SINet_Portrait_Segmentation)|SINet|EG1800, Baidu fashion dataset|humanseg|
|[FCN_HRNet_W18_Face_Seg](image/semantic_segmentation/FCN_HRNet_W18_Face_Seg)|FCN_HRNet_W18|-|humanseg|
|[ace2p](image/semantic_segmentation/ace2p)|ACE2P|LIP|ACE2P|
|[Pneumonia_CT_LKM_PP](image/semantic_segmentation/Pneumonia_CT_LKM_PP)|U-NET+|-|Pneumonia_CT|
|[Pneumonia_CT_LKM_PP_lung](image/semantic_segmentation/Pneumonia_CT_LKM_PP_lung)|U-NET+|-|Pneumonia_CT|
|[ocrnet_hrnetw18_voc](image/semantic_segmentation/ocrnet_hrnetw18_voc)|ocrnet, hrnet|PascalVoc2012|
|[U2Net](image/semantic_segmentation/U2Net)|U^2Net|-|U2Net|
|[U2Netp](image/semantic_segmentation/U2Netp)|U^2Net|-|U2Net|
|[Extract_Line_Draft](image/semantic_segmentation/Extract_Line_Draft)|UNet|Pixiv|Extract_Line_Draft|
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
|[pyramidbox_lite_mobile](image/face_detection/pyramidbox_lite_mobile)|PyramidBox|WIDER FACE|face_detection|
|[pyramidbox_lite_mobile_mask](image/face_detection/pyramidbox_lite_mobile_mask)|PyramidBox|WIDER FACE|face_detection|
|[pyramidbox_lite_server_mask](image/face_detection/pyramidbox_lite_server_mask)|PyramidBox|WIDER FACE|face_detection|
|[ultra_light_fast_generic_face_detector_1mb_640](image/face_detection/ultra_light_fast_generic_face_detector_1mb_640)|Ultra-Light-Fast-Generic-Face-Detector-1MB|WIDER FACE|face_detection|
|[ultra_light_fast_generic_face_detector_1mb_320](image/face_detection/ultra_light_fast_generic_face_detector_1mb_320)|Ultra-Light-Fast-Generic-Face-Detector-1MB|WIDER FACE|face_detection|
|[pyramidbox_lite_server](image/face_detection/pyramidbox_lite_server)|PyramidBox|WIDER FACE|face_detection|
|[pyramidbox_face_detection](image/face_detection/pyramidbox_face_detection)|PyramidBox|WIDER FACE|face_detection|

  - ### Text Recognition

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[chinese_ocr_db_crnn_mobile](image/text_recognition/chinese_ocr_db_crnn_mobile)|Differentiable Binarization+RCNN|icdar2015|Chinese text recognition|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/chinese_ocr_db_crnn_mobile) |
|[chinese_text_detection_db_mobile](image/text_recognition/chinese_text_detection_db_mobile)|Differentiable Binarization|icdar2015|Chinese text Detection|
|[chinese_text_detection_db_server](image/text_recognition/chinese_text_detection_db_server)|Differentiable Binarization|icdar2015|Chinese text Detection|
|[chinese_ocr_db_crnn_server](image/text_recognition/chinese_ocr_db_crnn_server)|Differentiable Binarization+RCNN|icdar2015|Chinese text recognition|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/chinese_ocr_db_crnn_server) |
|[Vehicle_License_Plate_Recognition](image/text_recognition/Vehicle_License_Plate_Recognition)|-|CCPD|Vehicle license plate recognition|
|[chinese_cht_ocr_db_crnn_mobile](image/text_recognition/chinese_cht_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Traditional Chinese text Detection|
|[japan_ocr_db_crnn_mobile](image/text_recognition/japan_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Japanese text recognition|
|[korean_ocr_db_crnn_mobile](image/text_recognition/korean_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Korean text recognition|
|[german_ocr_db_crnn_mobile](image/text_recognition/german_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|German text recognition|
|[french_ocr_db_crnn_mobile](image/text_recognition/french_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|French text recognition|
|[latin_ocr_db_crnn_mobile](image/text_recognition/latin_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Latin text recognition|
|[cyrillic_ocr_db_crnn_mobile](image/text_recognition/cyrillic_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Cyrillic text recognition|
|[multi_languages_ocr_db_crnn](image/text_recognition/multi_languages_ocr_db_crnn)|Differentiable Binarization+RCNN|icdar2015|Multi languages text recognition|
|[kannada_ocr_db_crnn_mobile](image/text_recognition/kannada_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Kannada text recognition|
|[arabic_ocr_db_crnn_mobile](image/text_recognition/arabic_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Arabic text recognition|
|[telugu_ocr_db_crnn_mobile](image/text_recognition/telugu_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Telugu text recognition|
|[devanagari_ocr_db_crnn_mobile](image/text_recognition/devanagari_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Devanagari text recognition|
|[tamil_ocr_db_crnn_mobile](image/text_recognition/tamil_ocr_db_crnn_mobile)|Differentiable Binarization+CRNN|icdar2015|Tamil text recognition|


  - ### Image Editing

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[realsr](image/Image_editing/super_resolution/realsr)|LP-KPN|RealSR dataset|Image / Video super-resolution|
|[deoldify](image/Image_editing/colorization/deoldify)|GAN|ILSVRC 2012|Black-and-white image / video colorization|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/deoldify) |
|[photo_restoration](image/Image_editing/colorization/photo_restoration)|deoldify + realsr|-|Old photo restoration|
|[user_guided_colorization](image/Image_editing/colorization/user_guided_colorization)|siggraph|ILSVRC 2012|User guided colorization|
|[falsr_c](image/Image_editing/super_resolution/falsr_c)|falsr_c| DIV2k|Lightweight super resolution - 2x|
|[dcscn](image/Image_editing/super_resolution/dcscn)|dcscn| DIV2k|Lightweight super resolution - 2x|
|[falsr_a](image/Image_editing/super_resolution/falsr_a)|falsr_a| DIV2k|Lightweight super resolution - 2x|
|[falsr_b](image/Image_editing/super_resolution/falsr_b)|falsr_b|DIV2k|Lightweight super resolution - 2x|

  - ### Instance Segmentation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[solov2](image/instance_segmentation/solov2)|-|COCO2014|Instance segmentation|

  - ### Object Detection

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[faster_rcnn_resnet50_coco2017](image/object_detection/faster_rcnn_resnet50_coco2017)|faster_rcnn|COCO2017||
|[ssd_vgg16_512_coco2017](image/object_detection/ssd_vgg16_512_coco2017)|SSD|COCO2017||
|[faster_rcnn_resnet50_fpn_venus](image/object_detection/faster_rcnn_resnet50_fpn_venus)|faster_rcnn|Baidu self built dataset|Large-scale general detection|
|[ssd_vgg16_300_coco2017](image/object_detection/ssd_vgg16_300_coco2017)||||
|[yolov3_resnet34_coco2017](image/object_detection/yolov3_resnet34_coco2017)|YOLOv3|COCO2017||
|[yolov3_darknet53_pedestrian](image/object_detection/yolov3_darknet53_pedestrian)|YOLOv3|Baidu Self built large-scale pedestrian dataset|Pedestrian Detection|
|[yolov3_mobilenet_v1_coco2017](image/object_detection/yolov3_mobilenet_v1_coco2017)|YOLOv3|COCO2017||
|[ssd_mobilenet_v1_pascal](image/object_detection/ssd_mobilenet_v1_pascal)|SSD|PASCAL VOC||
|[faster_rcnn_resnet50_fpn_coco2017](image/object_detection/faster_rcnn_resnet50_fpn_coco2017)|faster_rcnn|COCO2017||
|[yolov3_darknet53_coco2017](image/object_detection/yolov3_darknet53_coco2017)|YOLOv3|COCO2017||
|[yolov3_darknet53_vehicles](image/object_detection/yolov3_darknet53_vehicles)|YOLOv3|Baidu Self built large-scale vehicles dataset|vehicles Detection|
|[yolov3_darknet53_venus](image/object_detection/yolov3_darknet53_venus)|YOLOv3|Baidu self built datasetset|Large-scale general detection|
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
|[ernie_gen](text/text_generation/ernie_gen)|ERNIE-GEN|-|Pre-training finetuning framework for generating tasks|
|[ernie_gen_poetry](text/text_generation/ernie_gen_poetry)|ERNIE-GEN|Open source poetry dataset|Poetry generation|
|[ernie_gen_couplet](text/text_generation/ernie_gen_couplet)|ERNIE-GEN|Open source couplet dataset|Couplet generation|
|[ernie_gen_lover_words](text/text_generation/ernie_gen_lover_words)|ERNIE-GEN|Online love poems and love talk data|Love word generation|
|[ernie_tiny_couplet](text/text_generation/ernie_tiny_couplet)|Eernie_tiny|Open source couplet dataset|Couplet generation|
|[ernie_gen_acrostic_poetry](text/text_generation/ernie_gen_acrostic_poetry)|ERNIE-GEN|Open source poetry dataset|Acrostic poetry Generation|
|[Rumor_prediction](text/text_generation/Rumor_prediction)|-|Sina Weibo Chinese rumor data|Rumor prediction|
|[plato-mini](text/text_generation/plato-mini)|Unified Transformer|Billion level Chinese conversation data|Chinese dialogue|
|[plato2_en_large](text/text_generation/plato2_en_large)|plato2|Open domain multi round dataset|Super large scale generative dialogue|
|[plato2_en_base](text/text_generation/plato2_en_base)|plato2|Open domain multi round dataset|Super large scale generative dialogue|
|[CPM_LM](text/text_generation/CPM_LM)|GPT-2|Self built dataset|Chinese text generation|
|[unified_transformer-12L-cn](text/text_generation/unified_transformer-12L-cn)|Unified Transformer|Ten million level Chinese conversation data|Man machine multi round dialogue|
|[unified_transformer-12L-cn-luge](text/text_generation/unified_transformer-12L-cn-luge)|Unified Transformer|dialogue dataset|Man machine multi round dialogue|
|[reading_pictures_writing_poems](text/text_generation/reading_pictures_writing_poems)|Multi network cascade|-|Look at pictures and write poems|
|[GPT2_CPM_LM](text/text_generation/GPT2_CPM_LM)|||Q&A text generation|
|[GPT2_Base_CN](text/text_generation/GPT2_Base_CN)|||Q&A text generation|

  - ### Word Embedding

<details><summary>expand</summary><div>

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[w2v_weibo_target_word-bigram_dim300](text/embedding/w2v_weibo_target_word-bigram_dim300)|w2v|weibo||
|[w2v_baidu_encyclopedia_target_word-ngram_1-2_dim300](text/embedding/w2v_baidu_encyclopedia_target_word-ngram_1-2_dim300)|w2v|baidu_encyclopedia||
|[w2v_literature_target_word-word_dim300](text/embedding/w2v_literature_target_word-word_dim300)|w2v|literature||
|[word2vec_skipgram](text/embedding/word2vec_skipgram)|skip-gram|Baidu self built dataset||
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
|[roberta-wwm-ext-large](text/language_model/roberta-wwm-ext-large)|roberta-wwm-ext-large|Baidu self built dataset||
|[chinese-bert-wwm-ext](text/language_model/chinese_bert_wwm_ext)|chinese-bert-wwm-ext|Baidu self built dataset||
|[lda_webpage](text/language_model/lda_webpage)|LDA|Baidu Self built Web Page Domain Dataset||
|[lda_novel](text/language_model/lda_novel)||||
|[bert-base-multilingual-uncased](text/language_model/bert-base-multilingual-uncased)||||
|[rbt3](text/language_model/rbt3)||||
|[ernie_v2_eng_base](text/language_model/ernie_v2_eng_base)|ernie_v2_eng_base|Baidu self built dataset||
|[bert-base-multilingual-cased](text/language_model/bert-base-multilingual-cased)||||
|[rbtl3](text/language_model/rbtl3)||||
|[chinese-bert-wwm](text/language_model/chinese_bert_wwm)|chinese-bert-wwm|Baidu self built dataset||
|[bert-large-uncased](text/language_model/bert-large-uncased)||||
|[slda_novel](text/language_model/slda_novel)||||
|[slda_news](text/language_model/slda_news)||||
|[electra_small](text/language_model/electra_small)||||
|[slda_webpage](text/language_model/slda_webpage)||||
|[bert-base-cased](text/language_model/bert-base-cased)||||
|[slda_weibo](text/language_model/slda_weibo)||||
|[roberta-wwm-ext](text/language_model/roberta-wwm-ext)|roberta-wwm-ext|Baidu self built dataset||
|[bert-base-uncased](text/language_model/bert-base-uncased)||||
|[electra_large](text/language_model/electra_large)||||
|[ernie](text/language_model/ernie)|ernie-1.0|Baidu self built dataset||
|[simnet_bow](text/language_model/simnet_bow)|BOW|Baidu self built dataset||
|[ernie_tiny](text/language_model/ernie_tiny)|ernie_tiny|Baidu self built dataset||
|[bert-base-chinese](text/language_model/bert-base-chinese)|bert-base-chinese|Baidu self built dataset||
|[lda_news](text/language_model/lda_news)|LDA|Baidu Self built News Field Dataset||
|[electra_base](text/language_model/electra_base)||||
|[ernie_v2_eng_large](text/language_model/ernie_v2_eng_large)|ernie_v2_eng_large|Baidu self built dataset||
|[bert-large-cased](text/language_model/bert-large-cased)||||

</div></details>


  - ### Sentiment Analysis

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[ernie_skep_sentiment_analysis](text/sentiment_analysis/ernie_skep_sentiment_analysis)|SKEP|Baidu self built dataset|Sentence level sentiment analysis|
|[emotion_detection_textcnn](text/sentiment_analysis/emotion_detection_textcnn)|TextCNN|Baidu self built dataset|Dialogue emotion detection|
|[senta_bilstm](text/sentiment_analysis/senta_bilstm)|BiLSTM|Baidu self built dataset|Chinesesentiment analysis| [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/senta_bilstm) 
|[senta_bow](text/sentiment_analysis/senta_bow)|BOW|Baidu self built dataset|Chinese sentiment analysis|
|[senta_gru](text/sentiment_analysis/senta_gru)|GRU|Baidu self built dataset|Chinese sentiment analysis|
|[senta_lstm](text/sentiment_analysis/senta_lstm)|LSTM|Baidu self built dataset|Chinese sentiment analysis|
|[senta_cnn](text/sentiment_analysis/senta_cnn)|CNN|Baidu self built dataset|Chinese sentiment analysis|

  - ### Syntactic Analysis

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[DDParser](text/syntactic_analysis/DDParser)|Deep Biaffine Attention|Search query, web text, voice input and other data|Syntactic analysis|

  - ### Simultaneous Translation

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[transformer_nist_wait_1](text/simultaneous_translation/stacl/transformer_nist_wait_1)|transformer|NIST 2008|Chinese to English - wait-1|
|[transformer_nist_wait_3](text/simultaneous_translation/stacl/transformer_nist_wait_3)|transformer|NIST 2008|Chinese to English - wait-3|
|[transformer_nist_wait_5](text/simultaneous_translation/stacl/transformer_nist_wait_5)|transformer|NIST 2008|Chinese to English - wait-5|
|[transformer_nist_wait_7](text/simultaneous_translation/stacl/transformer_nist_wait_7)|transformer|NIST 2008|Chinese to English - wait-7|
|[transformer_nist_wait_all](text/simultaneous_translation/stacl/transformer_nist_wait_all)|transformer|NIST 2008|Chinese to English - waitk=-1|


  - ### Lexical Analysis

|module|Network|Dataset|Introduction|Huggingface Spaces Demo|
|--|--|--|--|--|
|[jieba_paddle](text/lexical_analysis/jieba_paddle)|BiGRU+CRF|Baidu self built dataset|Jieba uses Paddle to build a word segmentation network (two-way GRU). At the same time, it supports traditional word segmentation methods of jieba, such as precise mode, full mode, search engine mode, etc.|
|[lac](text/lexical_analysis/lac)|BiGRU+CRF|Baidu self built dataset|The lexical analysis model jointly developed by Baidu can complete the tasks of Chinese word segmentation, part of speech tagging and proper name recognition as a whole. Evaluated on Baidu self built dataset, LAC effect: Precision=88.0%, Recall=88.7%, F1 Score=88.4%.|[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PaddlePaddle/lac) 

  - ### Punctuation Restoration

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[auto_punc](text/punctuation_restoration/auto_punc)|Ernie-1.0|WuDaoCorpora 2.0|Automatically add 7 punctuation marks|

  - ### Text Review

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[porn_detection_cnn](text/text_review/porn_detection_cnn)|CNN|Baidu self built dataset|Pornography detection, automatically identify whether the text is pornographic and give the corresponding confidence, and identify pornographic descriptions, vulgar friends, and dirty documents in the text|
|[porn_detection_gru](text/text_review/porn_detection_gru)|GRU|Baidu self built dataset|Pornography detection, automatically identify whether the text is pornographic and give the corresponding confidence, and identify pornographic descriptions, vulgar friends, and dirty documents in the text|
|[porn_detection_lstm](text/text_review/porn_detection_lstm)|LSTM|Baidu self built dataset|Pornography detection, automatically identify whether the text is pornographic and give the corresponding confidence, and identify pornographic descriptions, vulgar friends, and dirty documents in the text|

## Audio

  - ### Voice cloning

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[ge2e_fastspeech2_pwgan](audio/voice_cloning/ge2e_fastspeech2_pwgan)|FastSpeech2|AISHELL-3|Chinese speech cloning|
|[lstm_tacotron2](audio/voice_cloning/lstm_tacotron2)|LSTM、Tacotron2、WaveFlow|AISHELL-3|Chinese speech cloning|

  - ### Text to Speech

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[transformer_tts_ljspeech](audio/tts/transformer_tts_ljspeech)|Transformer|LJSpeech-1.1|English speech synthesis|
|[fastspeech_ljspeech](audio/tts/fastspeech_ljspeech)|FastSpeech|LJSpeech-1.1|English speech synthesis|
|[fastspeech2_baker](audio/tts/fastspeech2_baker)|FastSpeech2|Chinese Standard Mandarin Speech Copus|Chinese speech synthesis|
|[fastspeech2_ljspeech](audio/tts/fastspeech2_ljspeech)|FastSpeech2|LJSpeech-1.1|English speech synthesis|
|[deepvoice3_ljspeech](audio/tts/deepvoice3_ljspeech)|DeepVoice3|LJSpeech-1.1|English speech synthesis|

  - ### Automatic Speech Recognition

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[deepspeech2_aishell](audio/asr/deepspeech2_aishell)|DeepSpeech2|AISHELL-1|Chinese Speech Recognition|
|[deepspeech2_librispeech](audio/asr/deepspeech2_librispeech)|DeepSpeech2|LibriSpeech|English Speech Recognition|
|[u2_conformer_aishell](audio/asr/u2_conformer_aishell)|Conformer|AISHELL-1|Chinese Speech Recognition|
|[u2_conformer_wenetspeech](audio/asr/u2_conformer_wenetspeech)|Conformer|WenetSpeech|Chinese Speech Recognition|
|[u2_conformer_librispeech](audio/asr/u2_conformer_librispeech)|Conformer|LibriSpeech|English Speech Recognition|


  - ### Audio Classification

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[panns_cnn6](audio/audio_classification/PANNs/cnn6)|PANNs|Google Audioset|It mainly includes 4 convolution layers and 2 full connection layers, and the model parameter is 4.5M. After pre-training, it can be used to extract the embbedding of audio. The dimension is 512|
|[panns_cnn14](audio/audio_classification/PANNs/cnn14)|PANNs|Google Audioset|It mainly includes 4 convolution layers and 2 full connection layers, and the model parameter is 4.5M. After pre-training, it can be used to extract the embbedding of audio. The dimension is 2048|
|[panns_cnn10](audio/audio_classification/PANNs/cnn10)|PANNs|Google Audioset|It mainly includes 4 convolution layers and 2 full connection layers, and the model parameter is 4.5M. After pre-training, it can be used to extract the embbedding of audio. The dimension is 512|

## Video
  - ### Video Classification

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[videotag_tsn_lstm](video/classification/videotag_tsn_lstm)|TSN + AttentionLSTM|Baidu self built dataset|Short-video classification|
|[tsn_kinetics400](video/classification/tsn_kinetics400)|TSN|Kinetics-400|Video classification|
|[tsm_kinetics400](video/classification/tsm_kinetics400)|TSM|Kinetics-400|Video classification|
|[stnet_kinetics400](video/classification/stnet_kinetics400)|StNet|Kinetics-400|Video classification|
|[nonlocal_kinetics400](video/classification/nonlocal_kinetics400)|Non-local|Kinetics-400|Video classification|


  - ### Video Editing

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[SkyAR](video/Video_editing/SkyAR)|UNet|UNet|Video sky Replacement|

  - ### Multiple Object tracking

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[fairmot_dla34](video/multiple_object_tracking/fairmot_dla34)|CenterNet|Caltech Pedestrian+CityPersons+CUHK-SYSU+PRW+ETHZ+MOT17|Realtime multiple object tracking|
|[jde_darknet53](video/multiple_object_tracking/jde_darknet53)|YOLOv3|Caltech Pedestrian+CityPersons+CUHK-SYSU+PRW+ETHZ+MOT17|object tracking with both accuracy and speed|

## Industrial Application

  - ### Meter Detection

|module|Network|Dataset|Introduction|
|--|--|--|--|
|[WatermeterSegmentation](image/semantic_segmentation/WatermeterSegmentation)|DeepLabV3|Water meter dataset|Water meter segmentation|
