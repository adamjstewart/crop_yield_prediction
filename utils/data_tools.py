"""Data tools for preprocessing the dataset."""

import pandas as pd


def drop_cols(data):
    """Drops columns we don't care about from the dataset.

    Parameters:
        data (pandas.DataFrame): the entire dataset
    """
    labels = [
        'County', 'State',
        'yield_irr', 'yield_noirr',
        'area_irr', 'area_noirr',
    ]

    data.drop(columns=labels, inplace=True)


def drop_nans(data):
    """Drops data points containing NaN values.

    Parameters:
        data (pandas.DataFrame): the entire dataset
    """
    data.dropna(inplace=True)


def encode_cols(data):
    """Converts categorical columns into indicator columns using
    a one-hot encoding.

    Parameters:
        data (pandas.DataFrame): the original dataset

    Returns:
        pandas.DataFrame: the transformed dataset
    """
    return pd.get_dummies(data, prefix=['FIPS'], columns=['FIPS'])


def split_dataset(data, start_train_year, end_train_year,
                  test_year, cross_validation):
    """Splits the dataset into training data and testing data.

    Parameters:
        data (pandas.DataFrame): the entire dataset
        start_train_year (int): the year to start training from
        end_train_year (int): the year to end training with
        test_year (int): the year to test on
        cross_validation (str): the cross-validation technique to perform

    Returns:
        pandas.DataFrame: the training data
        pandas.DataFrame: the testing data
    """
    # Only train on a subset of the data
    train_data = data[start_train_year <= data['year']]
    train_data = data[data['year'] <= end_train_year]

    if cross_validation == 'leave-one-out':
        # Train on every year except the test year
        train_data = train_data[train_data['year'] != test_year]
    elif cross_validation == 'forward':
        # Train on every year before the test year
        train_data = train_data[train_data['year'] < test_year]
    else:
        msg = "Unsupported cross_validation technique: '{}'"
        raise ValueError(msg.format(cross_validation))

    # Test on the test year
    test_data = data[data['year'] == test_year]

    return train_data, test_data


def array_to_series(predictions, index):
    """Converts an array of predictions to a Series.

    Parameters:
        predictions (numpy.ndarray): the predictions from the regressor
        index (numpy.ndarray): the index of the data

    Returns:
        pandas.Series: the predicted series data
    """
    return pd.Series(predictions, index)


def save_predictions(data, predictions, year):
    """Saves predicted values to the dataset.

    Parameters:
        data (pandas.DataFrame): the entire dataset
        predictions (pandas.Series): the predicted yields
        year (int): the year we are predicting
    """
    rows = data['year'] == year

    # Store the predictions in the DataFrame
    data.loc[rows, 'predicted yield'] = predictions
