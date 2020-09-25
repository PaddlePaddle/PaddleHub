import paddle
import paddlehub as hub

if __name__ == '__main__':
    place = paddle.CUDAPlace(0)
    paddle.disable_static()
    model = model = hub.Module(name='yolov3_darknet53_pascalvoc', is_train=False)
    model.eval()
    model.predict(imgpath="4026.jpeg", filelist="/PATH/TO/JSON/FILE")
