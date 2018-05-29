#!/usr/bin/env python3

"""
Deep learning model for crop yield prediction.

Written by Adam J. Stewart, 2018.
"""

from model import metrics
from model.regressor import *
from utils.data_tools import *
from utils.io_tools import *

import os
import sys
import tensorflow as tf


# Find the root directory of the project
ROOT_DIRECTORY = os.path.realpath(sys.path[0])


# Define command-line flags.
# Run `python crop_yield_prediction --help` for more details.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='model', default='linear',
    help="Model to use. Supports ['linear', 'dnn']"
)
flags.DEFINE_string(
    name='cross_validation', default='leave-one-out',
    help="Cross validation technique to perform. "
         "Supports ['leave-one-out', 'forward']"
)
flags.DEFINE_string(
    name='input_file',
    default=os.path.join(ROOT_DIRECTORY, 'data', 'Corn_model_data.csv'),
    help='Input dataset for yield prediction'
)
flags.DEFINE_string(
    name='output_file',
    default=os.path.join(ROOT_DIRECTORY, 'results', 'predictions.csv'),
    help='Output file to save results in'
)
flags.DEFINE_integer(
    name='buffer_size', default=100,
    help='Number of elements of the dataset to sample from'
)
flags.DEFINE_integer(
    name='num_epochs', default=1,
    help='Number of passes through the entire dataset'
)
flags.DEFINE_integer(
    name='batch_size', default=16,
    help='Number of samples in each mini-batch'
)
flags.DEFINE_boolean(
    name='verbose', default=True,
    help='Print messages explaining what is happening'
)


def main(args):
    """High-level pipeline.

    Trains the model and performs cross-validation.

    Parameters:
        args (list): command-line arguments
    """
    # Read in the dataset
    input_data = read_csv(FLAGS.input_file, FLAGS.verbose)

    # Filter data
    drop_cols(input_data)
    drop_nans(input_data)

    output_data = input_data.copy()

    # For each year...
    for year in get_years(input_data):
        if FLAGS.verbose:
            print('\nYear:', year)

        # Split the dataset into training and testing data
        train_data, test_data = split_dataset(
            input_data, year, FLAGS.cross_validation)

        # Initialize a new regression model
        model = get_regressor(FLAGS.model)

        # Train the model
        if FLAGS.verbose:
            print('\nTraining...')

        model.train(lambda: train_input_fn(
            train_data, FLAGS.buffer_size, FLAGS.num_epochs, FLAGS.batch_size))

        predictions = model.predict(lambda: test_input_fn(train_data))

        # Evaluate the performance
        labels = train_data['yield']
        predictions = generator_to_series(predictions, labels.index)

        rmse = metrics.rmse(labels, predictions)
        r = metrics.r(labels, predictions)
        R2 = metrics.R2(labels, predictions)

        if FLAGS.verbose:
            print('RMSE:', rmse)
            print('r:', r)
            print('R^2:', R2)

        # Test the model
        if FLAGS.verbose:
            print('Testing...\n')

        predictions = model.predict(lambda: test_input_fn(test_data))

        # Evaluate the performance
        labels = test_data['yield']
        predictions = generator_to_series(predictions, labels.index)

        rmse = metrics.rmse(labels, predictions)
        r = metrics.r(labels, predictions)
        R2 = metrics.R2(labels, predictions)

        if FLAGS.verbose:
            print('RMSE:', rmse)
            print('r:', r)
            print('R^2:', R2)

        save_predictions(output_data, predictions, year)

    # Calculate overall performance
    labels = input_data['yield']
    predictions = output_data['predicted yield']

    rmse = metrics.rmse(labels, predictions)
    r = metrics.r(labels, predictions)
    R2 = metrics.R2(labels, predictions)

    if FLAGS.verbose:
        print('\nOverall Performance\n')
        print('RMSE:', rmse)
        print('r:', r)
        print('R^2:', R2)

    # Write the resulting dataset
    write_csv(output_data, FLAGS.output_file, FLAGS.verbose)


if __name__ == '__main__':
    tf.app.run()
