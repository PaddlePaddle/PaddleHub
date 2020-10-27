import paddle
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='user_guided_colorization')
    state_dict = paddle.load('img_colorization_ckpt')
    model.set_dict(state_dict)
    result = model.predict('house.png')
