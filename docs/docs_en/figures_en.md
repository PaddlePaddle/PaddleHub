## Detailed Features

<a name="Various Pre-training Models"></a>

### 1\. Various Pre-training Models

#### 1.1. Image

|                      | **Examples of Boutique Models**                              |
| -------------------- | :----------------------------------------------------------- |
| Image Classification | [Dish Identification](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_dishes&en_category=ImageClassification), [Animal Identification](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification), [Animal Identification](https://www.paddlepaddle.org.cn/hubdetail?name=resnet50_vd_animals&en_category=ImageClassification), [-->More](../modules/image/classification/README.md) |
| Object Detection     | [Universal Detection](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_coco2017&en_category=ObjectDetection), [Pedestrian Detection](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_pedestrian&en_category=ObjectDetection), [Vehicle Detection](https://www.paddlepaddle.org.cn/hubdetail?name=yolov3_darknet53_vehicles&en_category=ObjectDetection), [-->More](../modules/image/object_detection/README.md) |
| Face Detection       | [Face Detection](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_server&en_category=FaceDetection), [Mask Detection](https://www.paddlepaddle.org.cn/hubdetail?name=pyramidbox_lite_server_mask&en_category=FaceDetection), [-->More](../modules/image/face_detection/README.md) |
| Image Segmentation   | [Portrait Segmentation](https://www.paddlepaddle.org.cn/hubdetail?name=deeplabv3p_xception65_humanseg&en_category=ImageSegmentation), [Body Analysis](https://www.paddlepaddle.org.cn/hubdetail?name=ace2p&en_category=ImageSegmentation), [Pneumonia CT Imaging Analysis](https://www.paddlepaddle.org.cn/hubdetail?name=Pneumonia_CT_LKM_PP&en_category=ImageSegmentation), [-->More](../modules/image/semantic_segmentation/README.md) |
| Key Point Detection  | [Body Key Points](https://www.paddlepaddle.org.cn/hubdetail?name=human_pose_estimation_resnet50_mpii&en_category=KeyPointDetection), [Face Key Points](https://www.paddlepaddle.org.cn/hubdetail?name=face_landmark_localization&en_category=KeyPointDetection), [Hands Key Points](https://www.paddlepaddle.org.cn/hubdetail?name=hand_pose_localization&en_category=KeyPointDetection), [-->More](./modules/image/keypoint_detection/README.md) |
| Text Recognition     | [Ultra Lightweight Chinese \& English OCR Text Recognition](https://www.paddlepaddle.org.cn/hubdetail?name=chinese_ocr_db_crnn_mobile&en_category=TextRecognition), [-->More](../modules/image/text_recognition/README.md) |
| Image Generation     | [Style Migration](https://www.paddlepaddle.org.cn/hubdetail?name=stylepro_artistic&en_category=GANs), [Street View Cartoon](https://www.paddlepaddle.org.cn/hubdetail?name=animegan_v2_hayao_99&en_category=GANs), [-->More](../modules/image/Image_gan/README.md) |
| Image Editing        | [Super Resolution](https://www.paddlepaddle.org.cn/hubdetail?name=realsr&en_category=ImageEditing), [B\&W Color](https://www.paddlepaddle.org.cn/hubdetail?name=deoldify&en_category=ImageEditing), [-->More](../modules/image/Image_editing/README.md) |

#### 1.2  Text

|                    | **Examples of Boutique Models**                              |
| ------------------ | :----------------------------------------------------------- |
| Word Analysis      | [Linguistic Analysis](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis), [Syntactic Analysis](https://www.paddlepaddle.org.cn/hubdetail?name=ddparser&en_category=SyntacticAnalysis), [-->More](../modules/text/lexical_analysis/README.md) |
| Sentiment Analysis | [Emotion Judgment](https://www.paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis), [Emotion Analysis](https://www.paddlepaddle.org.cn/hubdetail?name=emotion_detection_textcnn&en_category=SentimentAnalysis), [-->More](../modules/text/sentiment_analysis/README.md) |
| Text Review        | [Porn Review](https://www.paddlepaddle.org.cn/hubdetail?name=porn_detection_gru&en_category=TextCensorship), [-->More](../modules/text/text_review/README.md) |
| Text Generation    | [Poetic Couplet Generation](https://www.paddlepaddle.org.cn/hubdetail?name=ernie_tiny_couplet&en_category=TextGeneration), [Love Letters Generation](https://www.paddlepaddle.org.cn/hubdetail?name=ernie_gen_poetry&en_category=TextGeneration), [Popular Love Letters](https://www.paddlepaddle.org.cn/hubdetail?name=ernie_gen_lover_words&en_category=TextGeneration), [-->More](../modules/text/text_generation/README.md) |
| Semantic Models    | [ERNIE](https://www.paddlepaddle.org.cn/hubdetail?name=ERNIE&en_category=SemanticModel), [Text Similarity](https://www.paddlepaddle.org.cn/hubdetail?name=simnet_bow&en_category=SemanticModel), [-->More](../modules/text/language_model/README.md) |

#### 1.3. Speech

|                | **Examples of Boutique Models**                           |
| -------------- | :-------------------------------------------------------- |
| Text-to-speech | [Text-to-speech](https://www.paddlepaddle.org.cn/hubdetail?name=deepvoice3_ljspeech&en_category=TextToSpeech), [-->More](../modules/audio/README.md) |

#### 1.4. Video

|                      | **Examples of Boutique Models**                              |
| -------------------- | :----------------------------------------------------------- |
| Video Classification | [ Video Classification](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=VideoClassification), [-->More](../modules/video/README.md) |

<a name="One-key Model Prediction"></a>

### 2\. One-key Model Prediction

* For example, if you use the lightweight Chinese OCR model chinese\_ocr\_db\_crnn\_mobile for text recognition, you can quickly recognize the text in an image with pressing one key.

```shell
$ pip install paddlehub
$ wget https://paddlehub.bj.bcebos.com/model/image/ocr/test_ocr.jpg
$ hub run chinese_ocr_db_crnn_mobile --input_path test_ocr.jpg --visualization=True
```

* The prediction results images are stored in the ocr\_result folder under the current path, as shown in the following figure.

<p align="center">
 <img src="./imgs/ocr_res.jpg" width='70%' align="middle"  
</p>
* Use the lexical analysis model LAC for word segmentation.

```shell
$ hub run lac --input_text "现在，慕尼黑再保险公司不仅是此类行动的倡议者，更是将其大量气候数据整合进保险产品中，并与公众共享大量天气信息，参与到新能源领域的保障中。"
[{
    'word': ['现在', '，', '慕尼黑再保险公司', '不仅', '是', '此类', '行动', '的', '倡议者', '，', '更是', '将', '其', '大量', '气候', '数据', '整合', '进', '保险', '产品', '中', '，', '并', '与', '公众', '共享', '大量', '天气', '信息', '，', '参与', '到', '新能源', '领域', '的', '保障', '中', '。'],
    'tag':  ['TIME', 'w', 'ORG', 'c', 'v', 'r', 'n', 'u', 'n', 'w', 'd', 'p', 'r', 'a', 'n', 'n', 'v', 'v', 'n', 'n', 'f', 'w', 'c', 'p', 'n', 'v', 'a', 'n', 'n', 'w', 'v', 'v', 'n', 'n', 'u', 'vn', 'f', 'w']
}]
```

In addition to one-line code prediction, PaddleHub also supports the use of API to revoke the model. For details, refer to the detailed documentation of each model.

<a name="One-Key Model to Service"></a>

### 3\. One-Key to deploy Models as Services

PaddleHub provides convenient model-to-service capability to deploy HTTP services for models with one simple command. The LAC lexical analysis service can quickly start with the following commands:

```shell
$ hub serving start -m chinese_ocr_db_crnn_mobile
```

For more instructions on using Model Serving, See PaddleHub Model One-Key Model Serving Deployment.

<a name="Transfer Learning within ten lines of Codes"></a>

### 4\. Transfer Learning within Ten Lines of Codes

With the Fine-tune API, deep learning models can be migrated and learned in computer vision scenarios with a small number of codes.

* The [Demo Examples](../demo) provides rich codes for using Fine-tune API, including [Image Classification](../demo/image_classification), [Image Coloring](../demo/colorization), [Style Migration](../demo/style_transfer), and other scenario model migration examples.

<p align="center">
 <img src="../../imgs/paddlehub_finetune.gif" align="middle"  
</p>
<p align='center'>
 Transfer Learning within Ten Lines of Codes
</p>

* For a quick online experience, click [PaddleHub Tutorial Collection](https://aistudio.baidu.com/aistudio/projectdetail/231146) to use the GPU computing power provided by AI Studio platform for a quick attempt.
