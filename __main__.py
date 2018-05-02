#!/usr/bin/env python

"""
Deep learning model for crop yield prediction.

Written by Adam J. Stewart, 2018.
"""

from model.classifier import get_classifier
from utils.data_tools import filter_evi, filter_area
from utils.io_tools import read_csv, write_csv

import tensorflow as tf


# Define command-line flags.
# Run `python crop_yield_prediction --help` for more details.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='model', default='linear',
    help="Model to use. Supports ['linear', 'cnn']"
)
flags.DEFINE_string(
    name='loss', default='rmse',
    help="Loss metric to use. Supports ['rmse', 'r2']"
)
flags.DEFINE_string(
    name='cross_validation', default='leave-one-out',
    help="Cross validation technique to perform. "
         "Supports ['leave-one-out', 'forward']"
)
flags.DEFINE_string(
    name='input_file', default='data/Corn_model_data.csv',
    help='Input dataset for yield prediction'
)
flags.DEFINE_string(
    name='output_file', default='results/predictions.csv',
    help='Output file to save results in'
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
    data = read_csv(FLAGS.input_file, FLAGS.verbose)

    # Filter data
    data = filter_evi(data)
    data = filter_area(data)

    # Write the resulting dataset
    # write_csv(dataset, FLAGS.output_file, FLAGS.verbose)


if __name__ == '__main__':
    tf.app.run()
