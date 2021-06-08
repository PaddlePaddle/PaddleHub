import os
import math
import base64
from typing import Callable

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
matplotlib.use('Agg')


class HandDetect:
    """
    Detect hand pose information from body pose estimation result.

    Args:
        ratioWristElbow(float): Ratio to adjust the wrist center, ,default is 0.33.
    """

    def __init__(self, ratioWristElbow: float = 0.33):
        self.ratioWristElbow = ratioWristElbow

    def __call__(self, candidate: np.ndarray, subset: np.ndarray, oriImg: np.ndarray):
        detect_result = []
        image_height, image_width = oriImg.shape[0:2]
        for person in subset.astype(int):
            has_left = np.sum(person[[5, 6, 7]] == -1) == 0
            has_right = np.sum(person[[2, 3, 4]] == -1) == 0
            if not (has_left or has_right):
                continue
            hands = []
            # left hand
            if has_left:
                left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
                x1, y1 = candidate[left_shoulder_index][:2]
                x2, y2 = candidate[left_elbow_index][:2]
                x3, y3 = candidate[left_wrist_index][:2]
                hands.append([x1, y1, x2, y2, x3, y3, True])
            # right hand
            if has_right:
                right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
                x1, y1 = candidate[right_shoulder_index][:2]
                x2, y2 = candidate[right_elbow_index][:2]
                x3, y3 = candidate[right_wrist_index][:2]
                hands.append([x1, y1, x2, y2, x3, y3, False])

            for x1, y1, x2, y2, x3, y3, is_left in hands:

                x = x3 + self.ratioWristElbow * (x3 - x2)
                y = y3 + self.ratioWristElbow * (y3 - y2)
                distanceWristElbow = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
                distanceElbowShoulder = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)

                x -= width / 2
                y -= width / 2

                if x < 0: x = 0
                if y < 0: y = 0
                width1 = width
                width2 = width
                if x + width > image_width: width1 = image_width - x
                if y + width > image_height: width2 = image_height - y
                width = min(width1, width2)

                if width >= 20:
                    detect_result.append([int(x), int(y), int(width), is_left])

        return detect_result


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
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), self.stickwidth), int(angle), 0, 360,
                                           1)
                cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas


class DrawHandPose:
    """
        Draw hand pose estimation results on canvas.
        Args:
            show_number(bool): Whether to show estimation ids in canvas, default is False.

        """

    def __init__(self, show_number: bool = False):
        self.edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
                      [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        self.show_number = show_number

    def __call__(self, canvas: np.ndarray, all_hand_peaks: list):
        fig = Figure(figsize=plt.figaspect(canvas))

        fig.subplots_adjust(0, 0, 1, 1)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        bg = FigureCanvas(fig)
        ax = fig.subplots()
        ax.axis('off')
        ax.imshow(canvas)

        width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()

        for peaks in all_hand_peaks:
            for ie, e in enumerate(self.edges):
                if np.sum(np.all(peaks[e], axis=1) == 0) == 0:
                    x1, y1 = peaks[e[0]]
                    x2, y2 = peaks[e[1]]
                    ax.plot([x1, x2], [y1, y2],
                            color=matplotlib.colors.hsv_to_rgb([ie / float(len(self.edges)), 1.0, 1.0]))

            for i, keyponit in enumerate(peaks):
                x, y = keyponit
                ax.plot(x, y, 'r.')
                if self.show_number:
                    ax.text(x, y, str(i))
        bg.draw()
        canvas = np.frombuffer(bg.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return canvas


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


def npmax(array: np.ndarray):
    """Get max value and index."""
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


def cv2_to_base64(image: np.ndarray):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str: str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data
