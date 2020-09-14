import paddle
import paddlehub as hub

if __name__ == '__main__':
    place = paddle.CUDAPlace(0)
    paddle.disable_static()
    model = hub.Module(directory='msgnet')
    model.eval()
    result = model.predict("venice-boat.jpg", "candy.jpg")
