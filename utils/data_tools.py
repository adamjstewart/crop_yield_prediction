"""Data tools for preprocessing the dataset."""


def filter_core_states(data):
    """Filters data down to the 12 core states we care about.

    Parameters:
        data (pandas.DataFrame): the entire dataset

    Returns:
        pandas.DataFrame: the 12 core states
    """
    # List of states for which we have EVI data
    states = data.loc[data['evi6'].notna(), 'State']
    states = states.unique()

    # Filter data down to these 12 core states
    data = data.loc[data['State'].isin(states)]

    return data
