"""Input/output tools for reading and writing CSV files."""

import colorama
import os
import pandas as pd


# Metrics
RMSE = 'RMSE'
R2 = 'R2 (r * r)'
R2_CLASSIC = 'R2 (classic)'

METRICS = (RMSE, R2, R2_CLASSIC)


def read_dataset(filename, verbose=3):
    """Reads a CSV file containing the dataset.

    Parameters:
        filename (str): the filename of the dataset
        verbose (int): the verbosity level

    Returns:
        pandas.DataFrame: the dataset
    """
    if verbose > 1:
        print(colorama.Fore.BLUE + '\nReading {}...'.format(filename))

    return pd.read_csv(filename)


def write_dataset(data, output_dir, model, verbose=3):
    """Writes data to a CSV file.

    Parameters:
        data (pandas.DataFrame): the data to write
        output_dir (str): the directory to write to
        model (str): the machine learning model used
        verbose (int): the verbosity level
    """
    dirname = os.path.join(output_dir, model)
    filename = os.path.join(dirname, 'predictions.csv')

    if verbose > 1:
        print(colorama.Fore.BLUE + '\nWriting {}...'.format(filename))

    # Create directory if it does not already exist
    os.makedirs(dirname, exist_ok=True)

    # Write the CSV file
    data.to_csv(filename)


def write_performance(output_dir, model, ridge_lasso_alpha,
                      svr_kernel, svr_gamma, svr_c, svr_epsilon,
                      overall_stats, verbose=3):
    """Writes performance metrics to a CSV file.

    Parameters:
        output_dir (str): the directory to write to
        model (str): the machine learning model used
        ...
        verbose (int): the verbosity level
    """
    dirname = os.path.join(output_dir, model)
    filename = os.path.join(dirname, 'performance.csv')

    if verbose > 1:
        print(colorama.Fore.BLUE + '\nWriting {}...'.format(filename))

    # Create directory if it does not already exist
    os.makedirs(dirname, exist_ok=True)

    # Collect hyperparameters
    hyperparameters = []
    if model in ['ridge', 'lasso']:
        hyperparameters.append(ridge_lasso_alpha)
    elif model == 'svr':
        hyperparameters.extend([svr_kernel, svr_gamma, svr_c, svr_epsilon])

    # Collect statistics
    statistics = []
    for dataset in ('train', 'test'):
        types = ['median', 'mean']

        if dataset == 'test':
            types.append('combined')

        for type in types:
            for metric in METRICS:
                statistics.append(overall_stats[dataset][type][metric])

    # Write the CSV file
    with open(filename, 'a') as f:
        f.write(','.join(map(str, hyperparameters + statistics)) + '\n')
