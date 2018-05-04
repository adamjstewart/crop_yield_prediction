"""Data tools for preprocessing the dataset."""

import pandas as pd


def filter_evi(data):
    """Filters dataset by removing states without any EVI data.

    Parameters:
        data (pandas.DataFrame): the entire dataset

    Returns:
        pandas.DataFrame: the data for states with EVI data
    """
    # List of states for which we have EVI data
    states = data.loc[data['evi6'].notna(), 'State']
    states = states.unique()

    # Filter data down to these states
    data = data[data['State'].isin(states)]

    return data


def filter_area(data):
    """Filters dataset by removing counties without any growing area.

    Parameters:
        data (pandas.DataFrame): the entire dataset

    Returns:
        pandas.DataFrame: the data for counties with growing area
    """
    return data[data['area'].notna()]


def drop_cols(data):
    """Drops columns we don't care about from the dataset.

    Parameters:
        data (pandas.DataFrame): the entire dataset
    """
    labels = [
        'yield_irr', 'yield_noirr',
        'area_irr', 'area_noirr',
        'land_area',
    ]

    data.drop(columns=labels, inplace=True)


def drop_nans(data):
    """Drops data points containing NaN values.

    Parameters:
        data (pandas.DataFrame): the entire dataset
    """
    data.dropna(inplace=True)


def get_years(data):
    """Returns a list of years in the dataset in ascending order.

    Parameters:
        data (pandas.DataFrame): the entire dataset

    Returns:
        np.ndarray: the years present in the dataset
    """
    years = data['year'].unique()

    # Reverse order
    years = years[::-1]

    return years


def split_dataset(data, year, cross_validation):
    """Splits the dataset into training data and testing data.

    Parameters:
        data (pandas.DataFrame): the entire dataset
        year (int): the current year we are interested in
        cross_validation (str): the cross validation technique to perform.
            Supports ['leave-one-out', 'forward'].

    Returns:
        pandas.DataFrame: the training data
        pandas.DataFrame: the testing data
    """
    if cross_validation == 'leave-one-out':
        # Train on every year except the test year
        train_data = data[data['year'] != year]
    elif cross_validation == 'forward':
        # Train on every year before the test year
        train_data = data[data['year'] < year]
    else:
        msg = "cross_validation only supports 'leave-one-out' and 'forward'"
        raise ValueError(msg)

    # Test on the current year
    test_data = data[data['year'] == year]

    return train_data, test_data


def generator_to_series(predictions, index):
    """Converts a generator of dicts of arrays to a Series.

    Parameters:
        predictions (generator): the predictions from the regressor
        index (array): the index of the data

    Returns:
        pandas.Series: the predicted series data
    """
    data = [float(pred['predictions']) for pred in predictions]

    return pd.Series(data, index)


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
