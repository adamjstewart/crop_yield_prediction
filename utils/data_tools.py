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
    data = data.loc[data['State'].isin(states)]

    return data


def filter_area(data):
    """Filters dataset by removing counties without any growing area.

    Parameters:
        data (pandas.DataFrame): the entire dataset

    Returns:
        pandas.DataFrame: the data for counties with growing area
    """
    return data[data['area'].notna()]
