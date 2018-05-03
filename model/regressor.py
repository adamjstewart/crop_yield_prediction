"""Functions for initializing the regressor."""

import tensorflow as tf


def get_regressor(model):
    """Initializes a new regressor.

    Parameters:
        model (str): the regression model. Supports ['linear', 'dnn']

    Returns:
        tf.estimator.Estimator: the regressor
    """
    feature_columns = get_feature_columns()

    if model == 'linear':
        return tf.estimator.LinearRegressor(feature_columns)
    elif model == 'dnn':
        return tf.estimator.DNNRegressor([64, 32], feature_columns)
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
