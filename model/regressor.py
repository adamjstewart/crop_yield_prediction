"""Functions for initializing the regressor."""

import tempfile
import tensorflow as tf


def get_regressor(model):
    """Initializes a new regressor.

    Parameters:
        model (str): the regression model. Supports ['linear', 'dnn']

    Returns:
        tf.estimator.Estimator: the regressor
    """
    feature_columns = get_feature_columns()
    model_dir = tempfile.mkdtemp()

    if model == 'linear':
        return tf.estimator.LinearRegressor(feature_columns, model_dir)
    elif model == 'dnn':
        return tf.estimator.DNNRegressor([64, 32], feature_columns, model_dir)
    else:
        msg = "model only supports 'linear' and 'dnn'"
        raise ValueError(msg)


def get_feature_columns():
    """Returns a list of feature columns to train on.

    Returns:
        list: feature columns
    """
    features = [
        'year', 'FIPS', 'area',
        'tmax5', 'tmax6', 'tmax7', 'tmax8', 'tmax9',
        'tmin5', 'tmin6', 'tmin7', 'tmin8', 'tmin9',
        'tave5', 'tave6', 'tave7', 'tave8', 'tave9',
        'vpdmax5', 'vpdmax6', 'vpdmax7', 'vpdmax8', 'vpdmax9',
        'vpdmin5', 'vpdmin6', 'vpdmin7', 'vpdmin8', 'vpdmin9',
        'vpdave5', 'vpdave6', 'vpdave7', 'vpdave8', 'vpdave9',
        'precip5', 'precip6', 'precip7', 'precip8', 'precip9',
        'evi5', 'evi6', 'evi7', 'evi8', 'evi9',
        'lstmax5', 'lstmax6', 'lstmax7', 'lstmax8', 'lstmax9',
        'om', 'awc',
    ]

    return map(tf.feature_column.numeric_column, features)


def train_input_fn(data, buffer_size, num_epochs, batch_size):
    """Provides input data for training as mini-batches.

    Parameters:
        data (pandas.DataFrame): the entire dataset
        buffer_size (int): the number of elements of the dataset to sample from
        num_epochs (int): the number of passes through the entire dataset
        batch_size (int): the number of samples in each mini-batch

    Returns:
        tf.data.Dataset: the mini-batch to train on
    """
    # Separate the ground truth labels from the features
    features, labels = data, data.pop('yield')

    # Convert the data to a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Sample from a random subset of the data
    dataset = dataset.shuffle(buffer_size)

    # Repeat training on this subset
    dataset = dataset.repeat(num_epochs)

    # Sample a single mini-batch
    dataset = dataset.batch(batch_size)

    return dataset
