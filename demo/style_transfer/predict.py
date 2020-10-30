import paddle
import paddlehub as hub

if __name__ == '__main__':
    model = hub.Module(name='msgnet', load_checkpoint='/PATH/TO/CHECKPOINT')
    result = model.predict("venice-boat.jpg", "candy.jpg")
