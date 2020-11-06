# coding=utf-8
__all__ = ['RoIAlign']


class RoIAlign(object):
    def __init__(self, resolution=7, spatial_scale=0.0625, sampling_ratio=0):
        super(RoIAlign, self).__init__()
        if isinstance(resolution, int):
            resolution = [resolution, resolution]
        self.pooled_height = resolution[0]
        self.pooled_width = resolution[1]
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
