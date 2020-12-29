reading_pictures_writing_poems
类别 文本 - 文本生成

# 模型概述
看图写诗（reading_pictures_writing_poems），该模型可自动根据图像生成古诗词。该PaddleHub Module支持预测。

# 选择模型版本进行安装
$ hub install reading_pictures_writing_poems==1.0.0

# 命令行预测示例
$ hub run reading_pictures_writing_poems --input_image "scenery.jpg"

![](https://ai-studio-static-online.cdn.bcebos.com/69a9d5a5472449678a08e1ee5066c81b5859827647d74eb8a674afabbc205ae5)
<br>AI根据这张图片生成的古诗是： <br>
- 蕾蕾海河海，岳峰岳麓蔓。
- 不萌枝上春，自结心中线。

<br>
怎么样？还不错吧！
# Module API说明
## WritingPoem(self, image, use_gpu=False)
看图写诗预测接口，预测输入一张图像，输出一首古诗词
### 参数
- image(str): 待检测的图片路径
- use_gpu (bool): 是否使用 GPU
### 返回
- results (list[dict]): 识别结果的列表，列表中每一个元素为 dict，关键字有 image，Poetrys， 其中：
image字段为原输入图片的路径
Poetrys字段为输出的古诗词

# 代码示例
import paddlehub as hub

readingPicturesWritingPoems = hub.Module(directory="./reading_pictures_writing_poems")
readingPicturesWritingPoems.WritingPoem(image = "scenery.jpg", use_gpu=True)

# 贡献者
郑博培、彭兆帅

# 依赖
paddlepaddle >= 1.8.2
paddlehub >= 1.8.0
