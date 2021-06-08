import os
import paddle.nn as nn
from .skyfilter import SkyFilter
from paddlehub.module.module import moduleinfo


@moduleinfo(name="SkyAR", type="CV/Video_editing", author="jm12138", author_email="", summary="SkyAR", version="1.0.0")
class SkyAR(nn.Layer):
    def __init__(self, model_path=None):
        super(SkyAR, self).__init__()
        self.imgs = [
            'cloudy', 'district9ship', 'floatingcastle', 'galaxy', 'jupiter', 'rainy', 'sunny', 'sunset', 'supermoon'
        ]
        self.videos = ['thunderstorm']
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(self.directory, './ResNet50FCN')

    def MagicSky(self,
                 video_path,
                 save_path,
                 config='jupiter',
                 is_rainy=False,
                 preview_frames_num=0,
                 is_video_sky=False,
                 is_show=False,
                 skybox_img=None,
                 skybox_video=None,
                 rain_cap_path=None,
                 halo_effect=True,
                 auto_light_matching=False,
                 relighting_factor=0.8,
                 recoloring_factor=0.5,
                 skybox_center_crop=0.5):
        if config in self.imgs:
            skybox_img = os.path.join(self.directory, 'skybox', '%s.jpg' % config)
            skybox_video = None
            is_video_sky = False
        elif config in self.videos:
            skybox_img = None
            skybox_video = os.path.join(self.directory, 'skybox', '%s.mp4' % config)
            is_video_sky = True
        elif skybox_img:
            is_video_sky = False
            skybox_video = None
        elif is_video_sky and skybox_video:
            skybox_img = None
        else:
            raise 'please check your configs'

        if not rain_cap_path:
            rain_cap_path = os.path.join(self.directory, 'rain_streaks', 'videoplayback.mp4')

        skyfilter = SkyFilter(
            model_path=self.model_path,
            video_path=video_path,
            save_path=save_path,
            in_size=(384, 384),
            halo_effect=halo_effect,
            auto_light_matching=auto_light_matching,
            relighting_factor=relighting_factor,
            recoloring_factor=recoloring_factor,
            skybox_center_crop=skybox_center_crop,
            rain_cap_path=rain_cap_path,
            skybox_img=skybox_img,
            skybox_video=skybox_video,
            is_video=is_video_sky,
            is_rainy=is_rainy,
            is_show=is_show)

        skyfilter.run(preview_frames_num)
