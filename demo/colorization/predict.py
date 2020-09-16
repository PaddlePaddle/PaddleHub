import paddle
import paddlehub as hub
import paddle.nn as nn

if __name__ == '__main__':
    paddle.disable_static()
    model = hub.Module(name='user_guided_colorization')
    model.eval()
    result = model.predict(images='house.png')
