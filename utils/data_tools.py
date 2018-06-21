"""Data tools for preprocessing the dataset."""

from model.regressor import *

import pandas as pd
import sklearn


def drop_cols(data):
    """Drops columns we don't care about from the dataset.

    Parameters:
        data (pandas.DataFrame): the entire dataset
    """
    labels = [
        'County', 'State',
        'yield_irr', 'yield_noirr',
        'area', 'area_irr', 'area_noirr',
        'land_area',
    ]

    data.drop(labels, axis=1, inplace=True)


def drop_nans(data):
    """Drops data points containing NaN values.

    Parameters:
        data (pandas.DataFrame): the entire dataset
    """
    data.dropna(inplace=True)


def drop_unique(data):
    """Drops counties that only occur once in the dataset.

    It is not possible to learn a county fixed effect for these counties.

    Parameters:
        data (pandas.DataFrame): the entire dataset

    Returns:
        (pandas.DataFrame): the smaller dataset
    """
    return data[data.duplicated(subset=['FIPS'], keep=False)]


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

    return train_data.copy(), test_data.copy()


def shuffle(data):
    """Shuffles the dataset.

    Parameters:
        data (pandas.DataFrame): the original dataset

    Returns:
        pandas.DataFrame: the shuffled dataset
    """
    return sklearn.utils.shuffle(data)


def remove_annual_trend(train_data, test_data, jobs=-1):
    """Removes the annual yield trend from the dataset.

    Assumes a linear relationship between year and yield.

    Parameters:
        train_data (pandas.DataFrame): the training data
        test_data (pandas.DataFrame): the testing data
        jobs (int): the number of jobs to run in parallel

    Returns:
        pandas.DataFrame: the training data
        numpy.array: the training years
        pandas.DataFrame: the testing data
        numpy.array: the testing years
        LinearRegression: fitted regressor for annual trend
    """
    train_years = train_data.pop('year').values.reshape(-1, 1)
    train_yield = train_data['yield']

    model = get_linear_regressor(jobs)
    model.fit(train_years, train_yield)

    annual_trend = model.predict(train_years)
    annual_trend = array_to_series(annual_trend, train_yield.index)
    train_data['yield'] -= annual_trend

    test_years = test_data.pop('year').values.reshape(-1, 1)
    test_yield = test_data['yield']

    annual_trend = model.predict(test_years)
    annual_trend = array_to_series(annual_trend, test_yield.index)
    test_data['yield'] -= annual_trend

    return train_data, train_years, test_data, test_years, model


def reapply_annual_trend(labels, predictions, years, model):
    """Re-applies the annual yield trend.

    Parameters:
        labels (pandas.Series): the ground truth labels
        predictions (pandas.Series): the predicted labels
        years (pandas.Series): the corresponding years
        model (LinearRegression): fitted regressor for annual trend

    Returns:
        pandas.Series: the ground truth labels
        pandas.Series: the predicted labels
    """
    annual_trend = model.predict(years)

    labels += annual_trend
    predictions += annual_trend

    predictions = predictions.clip_lower(0)

    return labels, predictions


def remove_county_fixed_effect(train_data, test_data):
    """Removes the county fixed effect from the dataset.

    Parameters:
        train_data (pandas.DataFrame): the training data
        test_data (pandas.DataFrame): the testing data

    Returns:
        pandas.DataFrame: the training data
        pandas.Series: the FIPS county codes for the training data
        pandas.DataFrame: the testing data
        pandas.Series: the FIPS county codes for the testing data
        pandas.Series: the mean and std dev yields for each county
    """
    fips_groups = train_data.groupby('FIPS')
    county_fixed_effect = fips_groups['yield'].aggregate(['mean', 'std'])

    # Counties that only appear once in training set have NaN std dev
    county_fixed_effect['std'] = county_fixed_effect['std'].fillna(1)

    train_fips = train_data.pop('FIPS')
    test_fips = test_data.pop('FIPS')

    for index, fip in train_fips.iteritems():
        train_data.loc[index, 'yield'] -= county_fixed_effect.loc[fip, 'mean']
        train_data.loc[index, 'yield'] /= county_fixed_effect.loc[fip, 'std']

    for index, fip in test_fips.iteritems():
        test_data.loc[index, 'yield'] -= county_fixed_effect.loc[fip, 'mean']
        test_data.loc[index, 'yield'] /= county_fixed_effect.loc[fip, 'std']

    return train_data, train_fips, test_data, test_fips, county_fixed_effect


def reapply_county_fixed_effect(labels, predictions, fips,
                                county_fixed_effect):
    """Re-applies the county fixed effect.

    Parameters:
        labels (pandas.Series): the ground truth labels
        predictions (numpy.array): the predicted labels
        fips (pandas.Series): the FIPS county codes
        county_fixed_effect (pandas.Series): the mean and std dev yields
            for each county

    Returns:
        pandas.Series: the ground truth labels
        pandas.Series: the predicted labels
    """
    predictions = array_to_series(predictions, labels.index)

    for index, fip in fips.iteritems():
        labels.loc[index] *= county_fixed_effect.loc[fip, 'std']
        labels.loc[index] += county_fixed_effect.loc[fip, 'mean']
        predictions.loc[index] *= county_fixed_effect.loc[fip, 'std']
        predictions.loc[index] += county_fixed_effect.loc[fip, 'mean']

    return labels, predictions


def standardize(train_X, test_X):
    """Standardizes the dataset.

    Subtracts the mean and divides by the standard deviation of
    each numerical feature.

    Parameters:
        train_X (pandas.DataFrame): the training data
        test_X (pandas.DataFrame): the testing data

    Returns:
        train_X (pandas.DataFrame): the standardized training data
        test_X (pandas.DataFrame): the standardized testing data
    """
    scaler = sklearn.preprocessing.StandardScaler()

    # Compute the mean and standard deviation of the training set
    scaler.fit(train_X.loc[:, 'tmax5':'awc'])

    # Transform the training and testing sets
    train_X.loc[:, 'tmax5':'awc'] = scaler.transform(
        train_X.loc[:, 'tmax5':'awc'])
    test_X.loc[:, 'tmax5':'awc'] = scaler.transform(
        test_X.loc[:, 'tmax5':'awc'])

    return train_X, test_X


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
