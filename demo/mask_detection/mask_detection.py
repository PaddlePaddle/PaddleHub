# -*- coding:utf-8 -*-
import paddlehub as hub
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os

module = hub.Module(name="pyramidbox_lite_server_mask", version='1.1.0')


# opencv输出中文
def paint_chinese(im, chinese, position, fontsize, color_bgr):
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(
        'SourceHanSansSC-Medium.otf', fontsize, encoding="utf-8")
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    color = color_bgr[::-1]
    draw = ImageDraw.Draw(img_PIL)
    # PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    draw.text(position, chinese, font=font, fill=color)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片
    return img


result_path = './result'
if not os.path.exists(result_path):
    os.mkdir(result_path)

name = "./result/1-mask_detection.mp4"
width = 1280
height = 720
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'vp90')
writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

maskIndex = 0
index = 0
data = []

capture = cv2.VideoCapture(0)  # 打开摄像头
#capture = cv2.VideoCapture('./test_video.mp4')  # 打开视频文件
while True:
    frameData = {}
    ret, frame = capture.read()  # frame即视频的一帧数据
    if ret == False:
        break

    frame_copy = frame.copy()
    input_dict = {"data": [frame]}
    results = module.face_detection(data=input_dict)

    maskFrameDatas = []
    for result in results:
        label = result['data']['label']
        confidence_origin = result['data']['confidence']
        confidence = round(confidence_origin, 2)
        confidence_desc = str(confidence)

        top, right, bottom, left = int(result['data']['top']), int(
            result['data']['right']), int(result['data']['bottom']), int(
                result['data']['left'])

        #将当前帧保存为图片
        img_name = "avatar_%d.png" % (maskIndex)
        path = "./result/" + img_name
        image = frame[top - 10:bottom + 10, left - 10:right + 10]
        cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

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
        cv2.putText(frame_copy, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        #origin_point = (left, top - 36)
        #frame_copy = paint_chinese(frame_copy, label_cn, origin_point, 24,
        #                           color)

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

with open("./result/2-mask_detection.json", "w") as f:
    json.dump(data, f)

writer.release()

cv2.destroyAllWindows()
