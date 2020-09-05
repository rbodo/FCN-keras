import numpy as np


def score_prediction(y_true_one_hot, pred, num_classes):

    y_true = np.expand_dims(np.argmax(y_true_one_hot, -1), -1)
    y_pred = np.expand_dims(np.argmax(pred, -1), -1)

    dims = y_true.ndim - 1
    axes = tuple(np.arange(dims))
    classes = np.reshape(np.arange(num_classes), dims * [1] + [-1])
    positives = np.sum(np.equal(y_pred, classes), axes)
    true_num_pixels = np.sum(np.equal(y_true, classes), axes)
    true_positives = np.zeros(num_classes)
    for c in range(num_classes):
        class_mask = y_true == c
        true_positives[c] = np.sum(np.equal(y_true[class_mask],
                                            y_pred[class_mask]))

    pixel_accuracy = np.sum(true_positives) / np.sum(true_num_pixels)

    # Select only classes that are present.
    active_classes = np.flatnonzero(true_num_pixels)
    true_num_pixels = true_num_pixels[active_classes]
    true_positives = true_positives[active_classes]
    positives = positives[active_classes]

    correct_pixels_per_class = true_positives / true_num_pixels
    mean_accuracy = np.mean(correct_pixels_per_class)

    union = true_num_pixels + positives - true_positives
    IoU = true_positives / union
    mean_IoU = np.mean(IoU)

    freq_weighted_mean_IoU = (np.sum(true_num_pixels * IoU) /
                              np.sum(true_num_pixels))

    return pixel_accuracy, mean_accuracy, mean_IoU, freq_weighted_mean_IoU, \
        y_pred
