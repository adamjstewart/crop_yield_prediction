"""Input/output tools for reading and writing CSV files."""

import colorama
import pandas as pd
import pathlib


def read_csv(filename, verbose=False):
    """Reads a CSV file containing the dataset.

    Parameters:
        filename (str): the filename of the dataset
        verbose (bool): whether or not to print messages

    Returns:
        pandas.DataFrame: the dataset
    """
    if verbose:
        print(colorama.Fore.BLUE + '\nReading {}...'.format(filename))

    return pd.read_csv(filename)


def write_csv(data, filename, verbose=False):
    """Writes data to a CSV file.

    Parameters:
        data (pandas.DataFrame): the data to write
        filename (str): the filename to write to
        verbose (bool): whether or not to print messages
    """
    if verbose:
        print(colorama.Fore.BLUE + '\nWriting {}...'.format(filename))

    # Create directory if it does not already exist
    filename = pathlib.Path(filename)
    directory = filename.parent
    directory.mkdir(parents=True, exist_ok=True)

    # Write the CSV file
    data.to_csv(filename)
