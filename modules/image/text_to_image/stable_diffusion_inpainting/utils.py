import numpy as np
import paddle
import PIL
from PIL import Image


def preprocess(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = paddle.to_tensor(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)
    mask = 1 - mask  # repaint white, keep black
    mask = paddle.to_tensor(mask)
    return mask
