import cv2
import numpy as np
from sklearn.neighbors import KernelDensity

__all__ = [
    'build_transformation_matrix', 'update_transformation_matrix', 'estimate_partial_transform', 'removeOutliers',
    'guidedfilter'
]


def build_transformation_matrix(transform):
    """Convert transform list to transformation matrix

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


def update_transformation_matrix(M, m):

    # extend M and m to 3x3 by adding an [0,0,1] to their 3rd row
    M_ = np.concatenate([M, np.zeros([1, 3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1, 3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]


def estimate_partial_transform(matched_keypoints):
    """Wrapper of cv2.estimateRigidTransform for convenience in vidstab process

    :param matched_keypoints: output of match_keypoints util function; tuple of (cur_matched_kp, prev_matched_kp)
    :return: transform as list of [dx, dy, da]
    """
    prev_matched_kp, cur_matched_kp = matched_keypoints
    transform = cv2.estimateAffinePartial2D(np.array(prev_matched_kp), np.array(cur_matched_kp))[0]

    if transform is not None:
        # translation x
        dx = transform[0, 2]
        # translation y
        dy = transform[1, 2]
        # rotation
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0

    return [dx, dy, da]


def removeOutliers(prev_pts, curr_pts):

    d = np.sum((prev_pts - curr_pts)**2, axis=-1)**0.5

    d_ = np.array(d).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(d_)
    density = np.exp(kde.score_samples(d_))

    prev_pts = prev_pts[np.where((density >= 0.1))]
    curr_pts = curr_pts[np.where((density >= 0.1))]

    return prev_pts, curr_pts


def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0:r + 1, :] = imCum[r:2 * r + 1, :]
    imDst[r + 1:rows - r, :] = imCum[2 * r + 1:rows, :] - imCum[0:rows - 2 * r - 1, :]
    imDst[rows - r:rows, :] = np.tile(imCum[rows - 1, :], [r, 1]) - imCum[rows - 2 * r - 1:rows - r - 1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r + 1] = imCum[:, r:2 * r + 1]
    imDst[:, r + 1:cols - r] = imCum[:, 2 * r + 1:cols] - imCum[:, 0:cols - 2 * r - 1]
    imDst[:, cols - r:cols] = np.tile(imCum[:, cols - 1], [r, 1]).T - imCum[:, cols - 2 * r - 1:cols - r - 1]

    return imDst


def guidedfilter(img, p, r, eps):
    (rows, cols) = img.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(img, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(img * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(img * img, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * img + meanB
    return q
