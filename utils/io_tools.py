"""Input/output tools for reading and writing CSV files."""

import pandas as pd
import pathlib


def read_csv(filename):
    """Reads a CSV dataset.

    Parameters:
        filename (str): the filename of the dataset

    Returns:
        pandas.DataFrame: the dataset
    """
    return pd.read_csv(filename)


def write_csv(dataset, filename):
    """Writes a dataset to CSV.

    Parameters:
        dataset (pandas.DataFrame): the dataset to write
        filename (str): the filename to write to
    """
    # Create directory if it does not already exist
    directory = pathlib.Path(filename).parent
    directory.mkdir(parents=True, exist_ok=True)

    # Write the CSV file
    dataset.to_csv(filename)
