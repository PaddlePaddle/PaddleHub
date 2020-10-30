import paddle
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='user_guided_colorization')
    result = model.predict(images='house.png')
