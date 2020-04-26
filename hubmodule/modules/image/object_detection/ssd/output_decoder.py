class SSDOutputDecoder(object):
    # __op__ = fluid.layers.detection_output
    def __init__(self,
                 nms_threshold=0.3,
                 nms_top_k=400,
                 keep_top_k=200,
                 score_threshold=0.01,
                 nms_eta=1.0,
                 background_label=0):
        self.nms_threshold = nms_threshold
        self.background_label = background_label
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.score_threshold = score_threshold
        self.nms_eta = nms_eta
