#coding:utf-8
import os
import paddlehub as hub
import cv2

if __name__ == "__main__":
    ssd = hub.Module(name="ssd_mobilenet_v1_pascal")

    test_img_path = os.path.join("test", "test_img_bird.jpg")

    # execute predict and print the result
    results = ssd.object_detection(images=[cv2.imread(test_img_path)])
    for result in results:
        print(result)
