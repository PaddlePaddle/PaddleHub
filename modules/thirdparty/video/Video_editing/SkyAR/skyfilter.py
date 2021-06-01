import os
import cv2
import paddle
import numpy as np
from .skybox import SkyBox

__all__ = ['SkyFilter']


class SkyFilter():
    def __init__(self, model_path, video_path, save_path, in_size, halo_effect, auto_light_matching, relighting_factor,
                 recoloring_factor, skybox_center_crop, rain_cap_path, skybox_img, skybox_video, is_video, is_rainy,
                 is_show):
        self.in_size = in_size
        self.is_show = is_show
        self.cap = cv2.VideoCapture(video_path)
        self.m_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out_size = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.model = paddle.jit.load(model_path, model_filename='__model__', params_filename='__params__')
        self.model.eval()

        self.skyboxengine = SkyBox(
            out_size=self.out_size,
            skybox_img=skybox_img,
            skybox_video=skybox_video,
            halo_effect=halo_effect,
            auto_light_matching=auto_light_matching,
            relighting_factor=relighting_factor,
            recoloring_factor=recoloring_factor,
            skybox_center_crop=skybox_center_crop,
            rain_cap_path=rain_cap_path,
            is_video=is_video,
            is_rainy=is_rainy)
        path, _ = os.path.split(save_path)
        if path == '':
            path = '.'
        if not os.path.exists(path):
            os.mkdir(path)
        self.video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), self.fps, self.out_size)

    def synthesize(self, img_HD, img_HD_prev):
        h, w, _ = img_HD.shape

        img = cv2.resize(img_HD, self.in_size)
        img = np.array(img, dtype=np.float32)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
        img = paddle.to_tensor(img)

        G_pred = self.model(img)
        G_pred = paddle.nn.functional.interpolate(G_pred, (h, w), mode='bicubic', align_corners=False)
        G_pred = G_pred[0, :].transpose([1, 2, 0])
        G_pred = paddle.concat([G_pred, G_pred, G_pred], axis=-1)
        G_pred = G_pred.detach().numpy()
        G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

        skymask = self.skyboxengine.skymask_refinement(G_pred, img_HD)
        syneth = self.skyboxengine.skyblend(img_HD, img_HD_prev, skymask)

        return syneth, G_pred, skymask

    def run(self, preview_frames_num=0):

        img_HD_prev = None
        frames_num = preview_frames_num if 0 < preview_frames_num < self.m_frames else self.m_frames

        print('frames_num: %d, running evaluation...' % frames_num)
        for idx in range(1, frames_num + 1):
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.out_size)
                img_HD = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_HD = np.array(img_HD / 255., dtype=np.float32)

                if img_HD_prev is None:
                    img_HD_prev = img_HD

                syneth, _, _ = self.synthesize(img_HD, img_HD_prev)
                result = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
                self.video_writer.write(result)
                if self.is_show:
                    show_img = np.concatenate([frame, result], 1)
                    h, w = show_img.shape[:2]
                    show_img = cv2.resize(show_img, (720, int(720 / w * h)))
                    cv2.imshow('preview', show_img)
                    k = cv2.waitKey(1)
                    if (k == 27) or (cv2.getWindowProperty('preview', 0) == -1):
                        self.video_writer.release()
                        cv2.destroyAllWindows()
                        break
                print('processing: %d / %d ...' % (idx, frames_num))

                img_HD_prev = img_HD

            else:
                self.video_writer.release()
                cv2.destroyAllWindows()
                break
