import os
import base64
import math
from typing import Callable

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class PadDownRight:
    """
    Get padding images.

    Args:
        stride(int): Stride for calculate pad value for edges.
        padValue(int): Initialization for new area.
    """

    def __init__(self, stride: int = 8, padValue: int = 128):
        self.stride = stride
        self.padValue = padValue

    def __call__(self, img: np.ndarray):
        h, w = img.shape[0:2]
        pad = 4 * [0]
        pad[2] = 0 if (h % self.stride == 0) else self.stride - (h % self.stride)  # down
        pad[3] = 0 if (w % self.stride == 0) else self.stride - (w % self.stride)  # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1, :, :] * 0 + self.padValue, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:, 0:1, :] * 0 + self.padValue, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + self.padValue, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + self.padValue, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        return img_padded, pad


class RemovePadding:
    """
    Remove the padding values.

    Args:
        stride(int): Scales for resizing the images.

    """

    def __init__(self, stride: int = 8):
        self.stride = stride

    def __call__(self, data: np.ndarray, imageToTest_padded: np.ndarray, oriImg: np.ndarray, pad: list) -> np.ndarray:
        heatmap = np.transpose(np.squeeze(data), (1, 2, 0))
        heatmap = cv2.resize(heatmap, (0, 0), fx=self.stride, fy=self.stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        return heatmap


class GetPeak:
    """
    Get peak values and coordinate from input.

    Args:
        thresh(float): Threshold value for selecting peak value, default is 0.1.
    """

    def __init__(self, thresh=0.1):
        self.thresh = thresh

    def __call__(self, heatmap: np.ndarray):
        all_peaks = []
        peak_counter = 0
        for part in range(18):
            map_ori = heatmap[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down,
                 one_heatmap > self.thresh))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]], ) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i], ) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        return all_peaks


class Connection:
    """
    Get connection for selected estimation points.

    Args:
        mapIdx(list): Part Affinity Fields map index, default is None.
        limbSeq(list): Peak candidate map index, default is None.

    """

    def __init__(self, mapIdx: list = None, limbSeq: list = None):
        if mapIdx and limbSeq:
            self.mapIdx = mapIdx
            self.limbSeq = limbSeq
        else:
            self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                           [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                           [55, 56], [37, 38], [45, 46]]

            self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                            [1, 16], [16, 18], [3, 17], [6, 18]]
        self.caculate_vector = CalculateVector()

    def __call__(self, all_peaks: list, paf_avg: np.ndarray, orgimg: np.ndarray):
        connection_all = []
        special_k = []
        for k in range(len(self.mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in self.mapIdx[k]]]
            candA = all_peaks[self.limbSeq[k][0] - 1]
            candB = all_peaks[self.limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            if nA != 0 and nB != 0:
                connection_candidate = self.caculate_vector(candA, candB, nA, nB, score_mid, orgimg)
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        return connection_all, special_k


class CalculateVector:
    """
    Vector decomposition and normalization, refer Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
    for more details.

    Args:
        thresh(float): Threshold value for selecting candidate vector, default is 0.05.
    """

    def __init__(self, thresh: float = 0.05):
        self.thresh = thresh

    def __call__(self, candA: list, candB: list, nA: int, nB: int, score_mid: np.ndarray, oriImg: np.ndarray):
        connection_candidate = []
        for i in range(nA):
            for j in range(nB):
                vec = np.subtract(candB[j][:2], candA[i][:2])
                norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-5
                vec = np.divide(vec, norm)

                startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=10), \
                                    np.linspace(candA[i][1], candB[j][1], num=10)))

                vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                  for I in range(len(startend))])
                vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                  for I in range(len(startend))])

                score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                criterion1 = len(np.nonzero(score_midpts > self.thresh)[0]) > 0.8 * len(score_midpts)
                criterion2 = score_with_dist_prior > 0
                if criterion1 and criterion2:
                    connection_candidate.append(
                        [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
        return connection_candidate


class DrawPose:
    """
    Draw Pose estimation results on canvas.

    Args:
        stickwidth(int): Angle value to draw approximate ellipse curve, default is 4.

    """

    def __init__(self, stickwidth: int = 4):
        self.stickwidth = stickwidth

        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
                        [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                       [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
                       [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                       [255, 0, 170], [255, 0, 85]]

    def __call__(self, canvas: np.ndarray, candidate: np.ndarray, subset: np.ndarray):
        for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                x, y = candidate[index][0:2]
                cv2.circle(canvas, (int(x), int(y)), 4, self.colors[i], thickness=-1)
        for i in range(17):
            for n in range(len(subset)):
                index = subset[n][np.array(self.limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), self.stickwidth), \
                                           int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas


class Candidate:
    """
    Select candidate for body pose estimation.

    Args:
        mapIdx(list): Part Affinity Fields map index, default is None.
        limbSeq(list): Peak candidate map index, default is None.
    """

    def __init__(self, mapIdx: list = None, limbSeq: list = None):
        if mapIdx and limbSeq:
            self.mapIdx = mapIdx
            self.limbSeq = limbSeq
        else:
            self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                           [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                           [55, 56], [37, 38], [45, 46]]
            self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                            [1, 16], [16, 18], [3, 17], [6, 18]]

    def __call__(self, all_peaks: list, connection_all: list, special_k: list):
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        for k in range(len(self.mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(self.limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        return candidate, subset


class ResizeScaling:
    """Resize images by scaling method.

    Args:
        target(int): Target image size.
        interpolation(Callable): Interpolation method.
    """

    def __init__(self, target: int = 368, interpolation: Callable = cv2.INTER_CUBIC):
        self.target = target
        self.interpolation = interpolation

    def __call__(self, img, scale_search):
        scale = scale_search * self.target / img.shape[0]
        resize_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=self.interpolation)
        return resize_img


def cv2_to_base64(image: np.ndarray):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str: str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data
