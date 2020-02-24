# PaddleHub实现口罩佩戴检测应用

## 0 项目介绍
![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/BB6BC87A45D146CEBA7BF237B5383835?ynotemdtimestamp=1582271320612)

### [>点击查看视频链接<](https://www.bilibili.com/video/av88962128)

##### 背景
本项目可以部署在大型场馆出入口，学校，医院，交通通道出入口，人脸识别闸机，机器人上，支持的方案有：安卓方案（如RK3399的人脸识别机，机器人），ubuntu 边缘计算，windowsPC+摄像头，识别率80%~90%，如果立项使用场景可以达到 99% （如：人脸识别机场景）。但是限于清晰度和遮挡关系，对应用场景有一些要求。

##### 效果分析
可以看到识别率在80~90%之前，稍小的人脸有误识别的情况，有些挡住嘴的场景也被误识别成了戴口罩，一个人带着口罩，鼻子漏出来识别成没有戴口罩，这个是合理的因为的鼻子漏出来是佩戴不规范。初步判断，这个模型应用在门口，狭长通道，人脸识别机所在位置都是可以的。

![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/7E12DBD91D1D4AB5B33C84786D519065?ynotemdtimestamp=1582271320612)![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/2BD974FB990C4C448B30B04194545054?ynotemdtimestamp=1582271320612)![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/E49E34A071F8484D948511430FAB0360?ynotemdtimestamp=1582271320612)

## 1 部署环境
参考： https://www.paddlepaddle.org.cn/install/quick

### 安装paddlehub
`pip install paddlehub`


## 2 开发识别服务
### 加载预训练模型
```python
import paddlehub as hub
module = hub.Module(name="pyramidbox_lite_mobile_mask") #口罩检测模型
```

>以上语句paddlehub会自动下载口罩检测模型 "pyramidbox_lite_mobile_mask" 不需要提前下载模型

### OpenCV打开摄像头或视频文件
```python
import cv2

capture = cv2.VideoCapture(0) # 打开摄像头
# capture = cv2.VideoCapture('./2.mp4') # 打开视频文件
while(1):
    ret, frame = capture.read() # frame即视频的一帧数据

    if ret == False:
        break

    cv2.imshow('Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

### 口罩检测
```python
# frame为一帧数据

input_dict = {"data": [frame]}

results = module.face_detection(data=input_dict)

print(results)
```
输出结果：
```json
[
  {
    "data": {
      "label": "MASK",
      "left": 258.37087631225586,
      "right": 374.7980499267578,
      "top": 122.76758193969727,
      "bottom": 254.20085906982422,
      "confidence": 0.5630852
    },
    "id": 1
  }
]
```
>"label"：是否戴口罩，"confidence"：置信度，其余字段为脸框的位置大小

### 将结果显示到原视频帧中
```python
# results为口罩检测结果
for result in results:
    # print(result)

    label = result['data']['label']
    confidence = result['data']['confidence']

    top, right, bottom, left = int(result['data']['top']), int(result['data']['right']), int(result['data']['bottom']), int(result['data']['left'])

    color = (0, 255, 0)
    if label == 'NO MASK':
        color = (0, 0, 255)

    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
    cv2.putText(frame, label + ":" + str(confidence), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
```
![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/F85FCBCA17994C8691024381CBDAFCA7?ynotemdtimestamp=1582271320612)

>原DEMO中是英文+置信度显示在框的上面，尝试改为中文，遇到字体问题，以下是解决办法

### 图片写入中文
需要事先准备ttf/otf等格式的字体文件
```python
def paint_chinese_opencv(im,chinese,position,fontsize,color_bgr):#opencv输出中文
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))# 图像从OpenCV格式转换成PIL格式
    font = ImageFont.truetype('思源黑体SC-Heavy.otf',fontsize,encoding="utf-8") # 加载字体文件
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    color = color_bgr[::-1]
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=color)# PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)# PIL图片转cv2 图片
    return img
```
```python
for result in results:
    # print(result)

    label = result['data']['label']
    confidence = result['data']['confidence']

    top, right, bottom, left = int(result['data']['top']), int(result['data']['right']), int(result['data']['bottom']), int(result['data']['left'])

    color = (0, 255, 0)
    label_cn = "有口罩"
    if label == 'NO MASK':
        color = (0, 0, 255)
        label_cn = "无口罩"

    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
    # cv2.putText(frame, label + ":" + str(confidence), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    frame = paint_chinese_opencv(frame, label_cn + ":" + str(confidence), (left, top-36), 24, color)
```
![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/4F75E5C6F42F4C3CBE1341742D032847?ynotemdtimestamp=1582271320612)


### 提取头像文件
```python
img_name = "avatar_%d.png" % (maskIndex)
path = "./result/" + img_name
image = frame[top - 10: bottom + 10, left - 10: right + 10]
cv2.imwrite(path, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
```


### 结果写入JSON
```python
with open("./result/2-mask_detection.json","w") as f:
    json.dump(data, f)
```

>此处可以按照自己的应用需要改为输出到mysql，Redis，kafka ，MQ 供应用消化数据

### 完整代码如下
```python
import paddlehub as hub
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os

module = hub.Module(name="pyramidbox_lite_mobile_mask")


def paint_chinese_opencv(im,chinese,position,fontsize,color_bgr):#opencv输出中文
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))# 图像从OpenCV格式转换成PIL格式
    font = ImageFont.truetype('思源黑体SC-Heavy.otf',fontsize,encoding="utf-8") # 加载字体文件
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    color = color_bgr[::-1]
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=color)# PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)# PIL图片转cv2 图片
    return img


result_path = './result'
if not os.path.exists(result_path):
    os.mkdir(result_path)


name = "./result/1-mask_detection.mp4"
width = 1920
height = 1080
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

maskIndex = 0
index = 0
data = []

# capture = cv2.VideoCapture(0) # 打开摄像头
capture = cv2.VideoCapture('./2.mp4') # 打开视频文件
while(1):
    frameData = {}

    ret, frame = capture.read() # frame即视频的一帧数据

    if ret == False:
        break

    frame_copy = frame.copy()

    input_dict = {"data": [frame]}

    results = module.face_detection(data=input_dict)
    # print(results)

    maskFrameDatas = []
    for result in results:
        # print(result)

        label = result['data']['label']

        confidence_origin = result['data']['confidence']
        confidence = round(confidence_origin, 2)
        confidence_desc = str(confidence)

        top, right, bottom, left = int(result['data']['top']), int(result['data']['right']), int(result['data']['bottom']), int(result['data']['left'])

        #将当前帧保存为图片
        img_name = "avatar_%d.png" % (maskIndex)
        path = "./result/" + img_name
        image = frame[top - 10: bottom + 10, left - 10: right + 10]
        cv2.imwrite(path, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        maskFrameData = {}
        maskFrameData['top'] = top
        maskFrameData['right'] = right
        maskFrameData['bottom'] = bottom
        maskFrameData['left'] = left
        maskFrameData['confidence'] = float(confidence_origin)
        maskFrameData['label'] = label
        maskFrameData['img'] = img_name

        maskFrameDatas.append(maskFrameData)

        maskIndex += 1

        color = (0, 255, 0)
        label_cn = "有口罩"
        if label == 'NO MASK':
            color = (0, 0, 255)
            label_cn = "无口罩"

        cv2.rectangle(frame_copy, (left, top), (right, bottom), color, 3)
        # cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        frame_copy = paint_chinese_opencv(frame_copy, label_cn, (left, top-36), 24, color)


    writer.write(frame_copy)

    cv2.imshow('Mask Detection', frame_copy)

    frameData['frame'] = index
    # frameData['seconds'] = int(index/fps)
    frameData['data'] = maskFrameDatas

    data.append(frameData)

    print(json.dumps(frameData))

    index += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("./result/2-mask_detection.json","w") as f:
    json.dump(data, f)

writer.release()

cv2.destroyAllWindows()
```
## 3 制作网页呈现效果
此DEMO是显示一个固定视频，分析导出的 json 渲染到网页里面，如需实时显示需要再次开发

### python 导出的数据
使用上面的 python 文件完整执行后会有3个种类的数据输出，放到`web/video/result`目录下
![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/329AC9C2D89447EABE6B8C45D620441E?ynotemdtimestamp=1582271320612)

### json数据结构
![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/5D46F32061B047D4AB0AC016FE2A63A5?ynotemdtimestamp=1582271320612)

### 使用数据渲染网页

- 网页中左侧 "视频播放视频区"，播放同时实时回调当前播放的时间点
- 根据时间点换算为帧（1秒30帧），遍历 json 数据中的数据
- 把数据中对应的数据输出到网页右侧 "信息区"

![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/6329B326216A4950BF35E0CB37CDC58F?ynotemdtimestamp=1582271320612)

## 4 欢迎交流


**百度飞桨合作伙伴：**


![image](https://note.youdao.com/yws/public/resource/b0a4695bc7d58aed3b1ff797409aee1e/DC72DE1CF51747138871BB0E3D54E20D?ynotemdtimestamp=1582271320612)

北京奇想天外科技有限公司
