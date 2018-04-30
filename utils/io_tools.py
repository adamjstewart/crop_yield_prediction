"""Input/output tools for reading and writing CSV files."""

import pandas as pd
import pathlib


def read_csv(filename, verbose=False):
    """Reads a CSV dataset.

    Parameters:
        filename (str): the filename of the dataset
        verbose (bool): whether or not to print messages

    Returns:
        pandas.DataFrame: the dataset
    """
    if verbose:
        print('Reading {}...'.format(filename))

    return pd.read_csv(filename)


def write_csv(dataset, filename, verbose=False):
    """Writes a dataset to CSV.

    Parameters:
        dataset (pandas.DataFrame): the dataset to write
        filename (str): the filename to write to
        verbose (bool): whether or not to print messages
    """
    if verbose:
        print('Writing {}...'.format(filename))

    # Create directory if it does not already exist
    directory = pathlib.Path(filename).parent
    directory.mkdir(parents=True, exist_ok=True)

    # Write the CSV file
    dataset.to_csv(filename)
