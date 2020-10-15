import paddle
import paddlehub as hub

if __name__ == "__main__":

    paddle.disable_static()
    model = hub.Module(name='openpose_body_estimation')
    out1, out2 = model.predict("demo.jpg")
