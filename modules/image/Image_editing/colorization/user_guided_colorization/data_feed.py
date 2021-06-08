import paddle
import numpy as np


class ColorizeHint:
    """Get hint and mask images for colorization.

    This method is prepared for user guided colorization tasks. Take the original RGB images as imput,
    we will obtain the local hints and correspoding mask to guid colorization process.

    Args:
       percent(float): Probability for ignoring hint in an iteration.
       num_points(int): Number of selected hints in an iteration.
       samp(str): Sample method, default is normal.
       use_avg(bool): Whether to use mean in selected hint area.

    Return:
        hint(np.ndarray): hint images
        mask(np.ndarray): mask images
    """

    def __init__(self, percent: float, num_points: int = None, samp: str = 'normal', use_avg: bool = True):
        self.percent = percent
        self.num_points = num_points
        self.samp = samp
        self.use_avg = use_avg

    def __call__(self, data: np.ndarray, hint: np.ndarray, mask: np.ndarray):
        sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.data = data
        self.hint = hint
        self.mask = mask
        N, C, H, W = data.shape
        for nn in range(N):
            pp = 0
            cont_cond = True
            while cont_cond:
                if self.num_points is None:  # draw from geometric
                    # embed()
                    cont_cond = np.random.rand() < (1 - self.percent)
                else:  # add certain number of points
                    cont_cond = pp < self.num_points
                if not cont_cond:  # skip out of loop if condition not met
                    continue
                P = np.random.choice(sample_Ps)  # patch size
                # sample location
                if self.samp == 'normal':  # geometric distribution
                    h = int(np.clip(np.random.normal((H - P + 1) / 2., (H - P + 1) / 4.), 0, H - P))
                    w = int(np.clip(np.random.normal((W - P + 1) / 2., (W - P + 1) / 4.), 0, W - P))
                else:  # uniform distribution
                    h = np.random.randint(H - P + 1)
                    w = np.random.randint(W - P + 1)
                # add color point
                if self.use_avg:
                    # embed()
                    hint[nn, :, h:h + P, w:w + P] = np.mean(
                        np.mean(data[nn, :, h:h + P, w:w + P], axis=2, keepdims=True), axis=1, keepdims=True).reshape(
                            1, C, 1, 1)
                else:
                    hint[nn, :, h:h + P, w:w + P] = data[nn, :, h:h + P, w:w + P]
                mask[nn, :, h:h + P, w:w + P] = 1
                # increment counter
                pp += 1

        mask -= 0.5
        return hint, mask


class ColorizePreprocess:
    """Prepare dataset for image Colorization.

    Args:
       ab_thresh(float): Thresh value for setting mask value.
       p(float): Probability for ignoring hint in an iteration.
       num_points(int): Number of selected hints in an iteration.
       samp(str): Sample method, default is normal.
       use_avg(bool): Whether to use mean in selected hint area.
       is_train(bool): Training process or not.

    Return:
        data(dict)：The preprocessed data for colorization.

    """

    def __init__(self,
                 ab_thresh: float = 0.,
                 p: float = 0.,
                 points: int = None,
                 samp: str = 'normal',
                 use_avg: bool = True):
        self.ab_thresh = ab_thresh
        self.p = p
        self.num_points = points
        self.samp = samp
        self.use_avg = use_avg
        self.gethint = ColorizeHint(percent=self.p, num_points=self.num_points, samp=self.samp, use_avg=self.use_avg)

    def __call__(self, data_lab):
        """
        This method seperates the L channel and AB channel, obtain hint, mask and real_B_enc as the input for colorization task.

        Args:
           img(np.ndarray|paddle.Tensor): LAB image.

        Returns:
            data(dict)：The preprocessed data for colorization.
        """
        if type(data_lab) is not np.ndarray:
            data_lab = data_lab.numpy()
        data = {}
        A = 2 * 110 / 10 + 1
        data['A'] = data_lab[:, [0], :, :]
        data['B'] = data_lab[:, 1:, :, :]
        if self.ab_thresh > 0:  # mask out grayscale images
            thresh = 1. * self.ab_thresh / 110
            mask = np.sum(
                np.abs(np.max(np.max(data['B'], axis=3), axis=2) - np.min(np.min(data['B'], axis=3), axis=2)), axis=1)
            mask = (mask >= thresh)
            data['A'] = data['A'][mask, :, :, :]
            data['B'] = data['B'][mask, :, :, :]
            if np.sum(mask) == 0:
                return None
        data_ab_rs = np.round((data['B'][:, :, ::4, ::4] * 110. + 110.) / 10.)  # normalized bin number
        data['real_B_enc'] = data_ab_rs[:, [0], :, :] * A + data_ab_rs[:, [1], :, :]
        data['hint_B'] = np.zeros(shape=data['B'].shape)
        data['mask_B'] = np.zeros(shape=data['A'].shape)
        data['hint_B'], data['mask_B'] = self.gethint(data['B'], data['hint_B'], data['mask_B'])
        data['A'] = paddle.to_tensor(data['A'].astype(np.float32))
        data['B'] = paddle.to_tensor(data['B'].astype(np.float32))
        data['real_B_enc'] = paddle.to_tensor(data['real_B_enc'].astype(np.int64))
        data['hint_B'] = paddle.to_tensor(data['hint_B'].astype(np.float32))
        data['mask_B'] = paddle.to_tensor(data['mask_B'].astype(np.float32))
        return data
