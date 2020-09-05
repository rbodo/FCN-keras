import os
import numpy as np
from PIL import Image, ImageStat


def read_image(dataset_folder, filename, y_size, x_size):
    fpath = os.path.abspath(os.path.join(dataset_folder, filename))
    return np.array(Image.open(fpath).resize((x_size, y_size)))


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2

    factor = 1.

    img = img.crop(((x - w) * factor / zoom2, y - h / zoom2,
                    (x + w) * factor / zoom2, y + h / zoom2))

    return img


def convert_annotation_one_hot(annotation, num_classes):
    annotation_one_hot = np.zeros(annotation.shape + (num_classes,), bool)
    for c in range(num_classes):
        annotation_one_hot[annotation == c, c] = True

    return annotation_one_hot


def read_annotation(dataset_folder, filename, y_size, x_size):
    fpath = os.path.join(dataset_folder, filename)

    annotation = np.array(Image.open(fpath).resize((x_size, y_size)))

    annotation[annotation == 255] = 0

    return annotation


def normalize_0_mean_1_variance(img):
    img = Image.fromarray(img)

    stats = ImageStat.Stat(img)
    mean = stats.mean
    stddev = stats.stddev

    img = np.array(img)

    img = img - mean
    img = img / stddev

    return img


def normalize_0_1(img):
    img = img.astype('float')
    img /= 255.

    return img


def zoom_image(img):
    x_shift = 20

    img = img[x_shift:-x_shift, x_shift:-x_shift, :]

    return img
