"""Input/output tools for reading and writing CSV files."""

import pandas as pd


def read_csv(dataset):
    """Reads a CSV dataset.

    Parameters:
        dataset (str): the filename of the dataset

    Returns:
        pandas.DataFrame: the dataset
    """
    return pd.read_csv(dataset)
