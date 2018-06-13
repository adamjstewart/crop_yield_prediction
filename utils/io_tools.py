"""Input/output tools for reading and writing CSV files."""

import colorama
import os
import pandas as pd


def read_dataset(filename, verbose=False):
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


def write_dataset(data, output_dir, model, verbose=False):
    """Writes data to a CSV file.

    Parameters:
        data (pandas.DataFrame): the data to write
        output_dir (str): the directory to write to
        model (str): the machine learning model used
        verbose (bool): whether or not to print messages
    """
    dirname = os.path.join(output_dir, model)
    filename = os.path.join(dirname, 'predictions.csv')

    if verbose:
        print(colorama.Fore.BLUE + '\nWriting {}...'.format(filename))

    # Create directory if it does not already exist
    os.makedirs(dirname, exist_ok=True)

    # Write the CSV file
    data.to_csv(filename)


def write_performance(output_dir, model, ridge_lasso_alpha,
                      svr_c, svr_epsilon, svr_kernel,
                      median_training_rmse, median_training_r2,
                      median_training_r2_classic, mean_training_rmse,
                      mean_training_r2, mean_training_r2_classic,
                      combined_rmse, combined_r2, combined_r2_classic,
                      median_testing_rmse, median_testing_r2,
                      median_testing_r2_classic, mean_testing_rmse,
                      mean_testing_r2, mean_testing_r2_classic, verbose=False):
    """Writes performance metrics to a CSV file.

    Parameters:
        output_dir (str): the directory to write to
        model (str): the machine learning model used
        ...
        verbose (bool): whether or not to print messages
    """
    dirname = os.path.join(output_dir, model)
    filename = os.path.join(dirname, 'performance.csv')

    if verbose:
        print(colorama.Fore.BLUE + '\nWriting {}...'.format(filename))

    # Create directory if it does not already exist
    os.makedirs(dirname, exist_ok=True)

    # Write the CSV file
    with open(filename, 'a') as f:
        if model == 'svr':
            f.write(','.join(map(str, [
                svr_c, svr_epsilon, svr_kernel,
                median_training_rmse, median_training_r2,
                median_training_r2_classic, mean_training_rmse,
                mean_training_r2, mean_training_r2_classic,
                combined_rmse, combined_r2, combined_r2_classic,
                median_testing_rmse, median_testing_r2,
                median_testing_r2_classic, mean_testing_rmse,
                mean_testing_r2, mean_testing_r2_classic
            ])) + '\n')
