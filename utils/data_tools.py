"""Data tools for preprocessing the dataset."""


def filter_core_states(dataset):
    """Filters dataset down to the 12 core states we care about.

    Parameters:
        dataset (pandas.DataFrame): the entire dataset

    Returns:
        pandas.DataFrame: the 12 core states
    """
    return dataset[dataset['State'].isin(dataset.loc[dataset['evi6'].notnull(),'State'].unique())]
