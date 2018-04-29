"""Data tools for preprocessing the dataset."""

import numpy as np


def preprocess_labels(labels):
    """Preprocess labels to aid in classification.

    For our purposes, we only care about 3 labels:

    * corn - 1
    * soybean - 5
    * other - everything else

    Changes the classification of each label to:

    * corn - 1
    * soybean - 5
    * other - 0

    Args:
        labels (numpy.ndarray): the classification labels

    Returns:
        labels (numpy.ndarray): the labels with new classes
    """
    labels[(labels != 1) & (labels != 5)] = 0

    return labels


def sample_subset(data, labels, sample_size, rows, cols):
    """Take a random subset of the training data.

    Depending on your computational resources, the training data may be too
    large to train the classifier in a reasonable amount of time. This function
    returns a smaller subset of the training data of size sample_size.

    If sample_size is greater than the original image size or -1, returns the
    entire training dataset.

    Args:
        data (numpy.ndarray): the training data
        labels (numpy.ndarray): the training labels
        sample_size (int): the number of pixels in the subset
        rows (int): the number of rows
        cols (int): the number of columns

    Returns:
        data_subset (numpy.ndarray): a random subset of the data
        labels_subset (numpy.ndarray): a random subset of the labels
    """
    data_subset = data
    labels_subset = labels

    data_size = rows * cols
    if 0 < sample_size < data_size:
        index = np.random.choice(data_size, sample_size, replace=False)
        data_subset = data[index]
        labels_subset = labels[index]

    return data_subset, labels_subset
