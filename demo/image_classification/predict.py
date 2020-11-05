import paddle
import paddlehub as hub

if __name__ == '__main__':

    model = hub.Module(name='mobilenet_v2_imagenet', class_dim=5)
    state_dict = paddle.load('img_classification_ckpt')
    model.set_dict(state_dict)
    result = model.predict('flower.jpg')
