# coding=utf-8
class MultiBoxHead(object):
    # __op__ = fluid.layers.multi_box_head
    def __init__(self,
                 base_size,
                 num_classes,
                 aspect_ratios,
                 min_ratio=None,
                 max_ratio=None,
                 min_sizes=None,
                 max_sizes=None,
                 steps=None,
                 offset=0.5,
                 flip=True,
                 kernel_size=1,
                 pad=0,
                 min_max_aspect_ratios_order=False):
        self.base_size = base_size
        self.num_classes = num_classes
        self.aspect_ratios = aspect_ratios
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.steps = steps
        self.offset = offset
        self.flip = flip
        self.kernel_size = kernel_size
        self.pad = pad
        self.min_max_aspect_ratios_order = min_max_aspect_ratios_order
