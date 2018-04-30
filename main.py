#!/usr/bin/env python
#
# Deep learning model for corn yield prediction.
#
# Written by Adam J. Stewart, 2018.

from model.classifier import get_classifier
from utils.data_tools import preprocess_labels, sample_subset
from utils.io_tools import read_geotiff, write_geotiff

import tensorflow as tf


# Define command-line flags.
# Run `python main.py --help` for more details.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name='cross_validation', default='leave-one-out',
    help="Cross validation technique to perform. "
         "Supports ['leave-one-out', 'forward']"
)


def main(args):
    """High-level pipeline.

    Parameters:
        args (list): command-line arguments
    """
    pass


if __name__ == '__main__':
    tf.app.run()
