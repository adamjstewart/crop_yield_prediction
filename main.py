#!/usr/bin/env python
#
# Deep learning model for crop yield prediction.
#
# Written by Adam J. Stewart, 2018.

from model.classifier import get_classifier
from utils.data_tools import preprocess_labels, sample_subset
from utils.io_tools import read_csv

import tensorflow as tf


# Define command-line flags.
# Run `python main.py --help` for more details.
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
    name='dataset', default='data/Corn_model_data.csv',
    help='Dataset for yield prediction'
)


def main(args):
    """High-level pipeline.

    Trains the model and performs cross-validation.

    Parameters:
        args (list): command-line arguments
    """
    # Read in the dataset
    dataset = read_csv(FLAGS.dataset)


if __name__ == '__main__':
    tf.app.run()
