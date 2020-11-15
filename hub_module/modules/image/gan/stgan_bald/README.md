# stgan_bald
基于PaddleHub的秃头生成器
# 模型概述
秃头生成器（stgan_bald），该模型可自动根据图像生成1年、3年、5年的秃头效果。
# 模型效果：

详情请查看此链接：https://aistudio.baidu.com/aistudio/projectdetail/1145381

本模型为大家提供了小程序，欢迎大家体验

![image](https://github.com/1084667371/stgan_bald/blob/main/images/code.jpg)

# 选择模型版本进行安装
    $ hub install stgan_bald==1.0.0
# Module API说明
    def bald(self,
             images=None,
             paths=None,
             use_gpu=False,
             visualization=False):
秃头生成器API预测接口，预测输入一张人像，输出三张秃头效果(1年、3年、5年)
## 参数
    images (list(numpy.ndarray)): 图像数据，每个图像的形状为[H，W，C]，颜色空间为BGR。
    paths (list[str]): 图像的路径。
    use_gpu (bool): 是否使用gpu。
    visualization (bool): 是否保存图像。
## 返回
    data_0 ([numpy.ndarray]):秃头一年的预测结果图。
    data_1 ([numpy.ndarray]):秃头三年的预测结果图。
    data_2 ([numpy.ndarray]):秃头五年的预测结果图。
# API预测代码示例
    import paddlehub as hub
    import cv2
    
    stgan_bald = hub.Module('stgan_bald')
    im = cv2.imread('/PATH/TO/IMAGE')
    res = stgan_bald.bald(images=[im],visualization=True)
# 服务部署
## 第一步：启动PaddleHub Serving
$ hub serving start -m stgan_bald
## 第二步：发送预测请求
    import requests
    import json
    import base64

    import cv2
    import numpy as np

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')
    def base64_to_cv2(b64str):
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

    # 发送HTTP请求
    org_im = cv2.imread('/PATH/TO/IMAGE')
    data = {'images':[cv2_to_base64(org_im)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stgan_bald"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 保存图片 1年 3年 5年
    one_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_0']), cv2.COLOR_RGB2BGR)
    three_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_1']), cv2.COLOR_RGB2BGR)
    five_year =cv2.cvtColor(base64_to_cv2(r.json()["results"]['data_2']), cv2.COLOR_RGB2BGR)
    cv2.imwrite("segment_human_server.png", one_year)

# 贡献者
刘炫、彭兆帅、郑博培
# 依赖
paddlepaddle >= 1.8.2 

paddlehub >= 1.8.0

# 查看代码

[基于PaddleHub的秃头生成器](https://github.com/1084667371/PaddleHub/tree/release/v1.8/hub_module/modules/image/gan/stgan_bald)
