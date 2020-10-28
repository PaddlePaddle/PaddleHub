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
# API预测代码示例
    import paddlehub as hub
    import cv2
    stgan_bald = hub.Module(name="stgan_bald")
    result = stgan_bald.bald(images=[cv2.imread('/PATH/TO/IMAGE')])
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
    img0 = base64_to_cv2(r.json()["results"]['data_0'])
    img1 = base64_to_cv2(r.json()["results"]['data_1'])
    img2 = base64_to_cv2(r.json()["results"]['data_2'])
    img = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
    cv2.imwrite('bald_0.png', img)
# 贡献者
刘炫、彭兆帅、郑博培
# 依赖
paddlepaddle >= 1.8.2 

paddlehub >= 1.8.0

