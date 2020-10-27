import paddle
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='msgnet')
    state_dict = paddle.load('img_style_transfer_ckpt')
    model.set_dict(state_dict)
    result = model.predict("venice-boat.jpg", "candy.jpg")
