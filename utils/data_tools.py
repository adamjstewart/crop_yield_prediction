"""Data tools for preprocessing the dataset."""


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

    # Test on the current year
    test_data = data[data['year'] == year]

    return train_data, test_data
