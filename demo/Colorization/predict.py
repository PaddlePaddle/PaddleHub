import paddle.fluid as fluid
import paddlehub as hub
import paddle
import paddle.nn as nn

from hub_module.modules.image.Colorization.User_guided_colorization.module import Userguidedcolorization



if __name__ == '__main__':
    paddle.disable_static()
    model = Userguidedcolorization()
    model.eval()
    result = model.predict(images='/PATH/TO/IMAGE')


