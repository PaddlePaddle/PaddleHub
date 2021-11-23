import numpy as np
import paddle

from .pd_model.x2paddle_code import TFModel


class PosPrediction():
    def __init__(self, params, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = TFModel()
        self.network.set_dict(params, use_structured_name=False)
        self.network.eval()

    def predict(self, image):
        paddle.disable_static()
        image_tensor = paddle.to_tensor(image[np.newaxis, :, :, :], dtype='float32')
        pos = self.network(image_tensor)
        pos = pos.numpy()
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, feed_dict={self.x: images})
        return pos * self.MaxPos
