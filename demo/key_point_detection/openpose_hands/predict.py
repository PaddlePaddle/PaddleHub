import paddle
import paddlehub as hub

if __name__ == "__main__":

    paddle.disable_static()
    model = hub.Module(name='openpose_hands_estimation')
    all_hand_peaks = model.predict("demo.jpg")
