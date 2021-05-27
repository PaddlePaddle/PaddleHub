import paddle
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='ocrnet_hrnetw18_voc', num_classes=2, pretrained='/PATH/TO/CHECKPOINT')
    model.predict(images=["N0007.jpg"], visualization=True)
