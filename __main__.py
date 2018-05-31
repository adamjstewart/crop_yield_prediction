#!/usr/bin/env python3

"""
Machine learning model for crop yield prediction.

Written by Adam J. Stewart, 2018.
"""

from model import metrics
from model.regressor import *
from utils.data_tools import *
from utils.io_tools import *

import argparse
import colorama
import os
import sys


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter,
                          argparse.ArgumentDefaultsHelpFormatter):
    """Custom argparse help formatter.

    Allows for newline characters in description.
    Automatically adds argument defaults.
    """


def set_up_parser():
    """Sets up the argument parser.

    Returns:
        argparse.ArgumentParser: the argument parser
    """
    # Find the root directory of the project
    ROOT_DIRECTORY = os.path.realpath(sys.path[0])

    # Initialize new parser
    parser = argparse.ArgumentParser(
        prog='crop_yield_prediction',
        description=__doc__,
        formatter_class=CustomHelpFormatter)

    # Add arguments to the parser
    parser.add_argument(
        '-i', '--input-file',
        default=os.path.join(ROOT_DIRECTORY, 'data', 'Corn_model_data.csv'),
        help='input dataset for yield prediction')
    parser.add_argument(
        '-o', '--output-file',
        default=os.path.join(ROOT_DIRECTORY, 'results', 'predictions.csv'),
        help='output file to save results in')
    parser.add_argument(
        '-m', '--model',
        default='linear',
        choices=['linear', 'ridge', 'svr', 'random-forest'],
        help='regression model to use')
    parser.add_argument(
        '--cross-validation',
        default='leave-one-out',
        choices=['leave-one-out', 'forward'],
        help='cross-validation technique to perform')
    parser.add_argument(
        '--start-train-year',
        default=2003, type=int,
        help='year to start training from')
    parser.add_argument(
        '--end-train-year',
        default=2016, type=int,
        help='year to end training with')
    parser.add_argument(
        '--start-test-year',
        default=2003, type=int,
        help='year to start testing from')
    parser.add_argument(
        '--end-test-year',
        default=2016, type=int,
        help='year to end testing with')
    parser.add_argument(
        '-a', '--alpha',
        default=1.0, type=float,
        help='regularization strength')
    parser.add_argument(
        '-c',
        default=1.0, type=float,
        help='SVR penalty parameter')
    parser.add_argument(
        '-e', '--epsilon',
        default=0.1, type=float,
        help='SVR epsilon')
    parser.add_argument(
        '-v', '--verbose',
        default=3, type=int,
        help='verbosity level')
    parser.add_argument(
        '-j', '--jobs',
        default=-1, type=int,
        help='number of jobs to run in parallel')

    return parser


def main(args):
    """High-level pipeline.

    Trains the model and performs cross-validation.

    Parameters:
        args (argparse.Namespace): command-line arguments
    """
    # Read in the dataset
    input_data = read_csv(args.input_file, args.verbose)

    # Data preprocessing
    drop_cols(input_data)
    drop_nans(input_data)

    output_data = input_data.copy()

    input_data = encode_cols(input_data)

    # Initialize a new regression model
    model = get_regressor(
        args.model, args.alpha, args.c, args.epsilon,
        args.verbose, args.jobs)

    # For each testing year...
    for test_year in range(args.start_test_year, args.end_test_year + 1):
        if args.verbose:
            print(colorama.Fore.RED + '\nYear:', test_year)

        # Split the dataset into training and testing data
        train_data, test_data = split_dataset(
            input_data, args.start_train_year, args.end_train_year,
            test_year, args.cross_validation)

        train_X, train_y = train_data, train_data.pop('yield')
        test_X, test_y = test_data, test_data.pop('yield')

        # Train the model
        if args.verbose:
            print(colorama.Fore.GREEN + '\nTraining...\n')

        model.fit(train_X, train_y)

        predictions = model.predict(train_X)
        predictions = array_to_series(predictions, train_y.index)

        # Evaluate the performance
        rmse = metrics.rmse(train_y, predictions)
        r = metrics.r(train_y, predictions)
        R2 = metrics.R2(train_y, predictions)

        if args.verbose:
            print(colorama.Fore.CYAN + 'RMSE:', rmse)
            print(colorama.Fore.CYAN + 'r:', r)
            print(colorama.Fore.CYAN + 'R^2:', R2)

        # Test the model
        if args.verbose:
            print(colorama.Fore.GREEN + '\nTesting...\n')

        predictions = model.predict(test_X)
        predictions = array_to_series(predictions, test_y.index)

        # Evaluate the performance
        rmse = metrics.rmse(test_y, predictions)
        r = metrics.r(test_y, predictions)
        R2 = metrics.R2(test_y, predictions)

        if args.verbose:
            print(colorama.Fore.CYAN + 'RMSE:', rmse)
            print(colorama.Fore.CYAN + 'r:', r)
            print(colorama.Fore.CYAN + 'R^2:', R2)

        save_predictions(output_data, predictions, test_year)

    # Calculate overall performance
    labels = input_data['yield']
    predictions = output_data['predicted yield']

    rmse = metrics.rmse(labels, predictions)
    r = metrics.r(labels, predictions)
    R2 = metrics.R2(labels, predictions)

    if args.verbose:
        print(colorama.Fore.RED + '\nOverall Performance\n')

        print(colorama.Fore.CYAN + 'RMSE:', rmse)
        print(colorama.Fore.CYAN + 'r:', r)
        print(colorama.Fore.CYAN + 'R^2:', R2)

    # Write the resulting dataset
    write_csv(output_data, args.output_file, args.verbose)


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    if args.verbose:
        colorama.init(autoreset=True)

    main(args)
