import cv2
import numpy as np
import six
import os
import glob


def resize_short(img, target_size, interpolation=None):
    """resize image

    Args:
        img: image data
        target_size: resize short target size
        interpolation: interpolation mode

    Returns:
        resized image data
    """
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    if interpolation:
        resized = cv2.resize(
            img, (resized_width, resized_height), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    """crop image

    Args:
        img: images data
        target_size: crop target size
        center: crop mode

    Returns:
        img: cropped image data
    """
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def preprocess_image(img, random_mirror=True):
    """
    centered, scaled by 1/255.
    :param img: np.array: shape: [ns, h, w, 3], color order: rgb.
    :return: np.array: shape: [ns, h, w, 3]
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transpose to [ns, 3, h, w]
    img = img.astype('float32').transpose((0, 3, 1, 2)) / 255

    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    if random_mirror:
        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            img = img[:, :, ::-1, :]

    return img


def _find_classes(dir):
    # Faster and available in Python 3.5 and above
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class ReaderConfig():
    """
    A generic data loader where the images are arranged in this way:

        root/train/dog/xxy.jpg
        root/train/dog/xxz.jpg
        ...
        root/train/cat/nsdf3.jpg
        root/train/cat/asd932_.jpg
        ...

        root/test/dog/xxx.jpg
        ...
        root/test/cat/123.jpg
        ...

    """

    def __init__(self, dataset_dir, is_test):
        image_paths, labels, self.num_classes = self.reader_creator(
            dataset_dir, is_test)
        random_per = np.random.permutation(range(len(image_paths)))
        self.image_paths = image_paths[random_per]
        self.labels = labels[random_per]
        self.is_test = is_test

    def get_reader(self):
        def reader():
            IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                              '.tif', '.tiff', '.webp')
            target_size = 256
            crop_size = 224

            for i, img_path in enumerate(self.image_paths):
                if not img_path.lower().endswith(IMG_EXTENSIONS):
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(img_path)
                    continue
                img = resize_short(img, target_size, interpolation=None)
                img = crop_image(img, crop_size, center=self.is_test)
                img = img[:, :, ::-1]
                img = np.expand_dims(img, axis=0)

                img = preprocess_image(img, not self.is_test)

                yield img, self.labels[i]

        return reader

    def reader_creator(self, dataset_dir, is_test=False):
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                          '.tif', '.tiff', '.webp')

        # read
        if is_test:
            datasubset_dir = os.path.join(dataset_dir, 'test')
        else:
            datasubset_dir = os.path.join(dataset_dir, 'train')

        class_names, class_to_idx = _find_classes(datasubset_dir)
        # num_classes = len(class_names)
        image_paths = []
        labels = []
        for class_name in class_names:
            classes_dir = os.path.join(datasubset_dir, class_name)
            for img_path in glob.glob(os.path.join(classes_dir, '*')):
                if not img_path.lower().endswith(IMG_EXTENSIONS):
                    continue

                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])

        image_paths = np.array(image_paths)
        labels = np.array(labels)
        return image_paths, labels, len(class_names)
