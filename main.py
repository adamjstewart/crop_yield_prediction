#!/usr/bin/env python
#
# Machine learning classifier for satellite imagery
#
# Written by Adam J. Stewart, 2018

from model.classifier import get_classifier
from utils.data_tools import preprocess_labels, sample_subset
from utils.io_tools import read_geotiff, write_geotiff

import argparse


def setup_parser():
    """Sets up the argument parser and returns it."""

    desc = 'Machine learning classifier for satellite imagery'

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--train-data',
        default='data/20130824_RE3_3A_Analytic_Champaign_north.tif',
        help='GeoTIFF file containing training data [default: %(default)s]')

    parser.add_argument(
        '--train-labels',
        default='data/CDL_2013_Champaign_north.tif',
        help='GeoTIFF file containing training labels [default: %(default)s]')

    parser.add_argument(
        '--test-data',
        default='data/20130824_RE3_3A_Analytic_Champaign_south.tif',
        help='GeoTIFF file containing testing data [default: %(default)s]')

    parser.add_argument(
        '--test-labels',
        default='data/CDL_2013_Champaign_south.tif',
        help='GeoTIFF file containing testing labels [default: %(default)s]')

    parser.add_argument(
        '-c', '--classifier',
        choices=['logistic', 'svm', 'random_forest'],
        default='logistic',
        help='classifier to use [default: %(default)s]')

    parser.add_argument(
        '-j', '--jobs',
        type=int, default=-1,
        help='number of jobs to run in parallel [default: # of cores]')

    parser.add_argument(
        '-s', '--sample-size',
        type=int, default=100000,
        help='number of randomly sampled points from training data '
             '[default: %(default)s]')

    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='save intermediate files for debugging purposes')

    parser.add_argument(
        '-v', '--verbose',
        type=int, default=3,
        help='verbosity level [default: %(default)s]')

    return parser


if __name__ == '__main__':
    # Parse supplied arguments
    parser = setup_parser()
    args = parser.parse_args()

    if args.verbose > 0:
        print('\nReading input data...\n')

    # Read the training dataset
    (
        train_data_dataset,
        train_data_array,
        train_data_geo_transform,
        train_data_projection,
        train_data_ctable,
        train_data_rows,
        train_data_cols
    ) = read_geotiff(args.train_data, args.verbose)
    (
        train_labels_dataset,
        train_labels_array,
        train_labels_geo_transform,
        train_labels_projection,
        train_labels_ctable,
        train_labels_rows,
        train_labels_cols
    ) = read_geotiff(args.train_labels, args.verbose)

    # Read the testing dataset
    (
        test_data_dataset,
        test_data_array,
        test_data_geo_transform,
        test_data_projection,
        test_data_ctable,
        test_data_rows,
        test_data_cols
    ) = read_geotiff(args.test_data, args.verbose)

    # Preprocess the training labels
    train_labels_array = preprocess_labels(train_labels_array)

    # Save preprocessed labels for debugging purposes
    if args.debug:
        write_geotiff(
            train_labels_array,
            train_labels_geo_transform,
            train_labels_projection,
            train_labels_ctable,
            train_labels_rows,
            train_labels_cols,
            args.train_labels,
            suffixes=['preprocessed'],
            verbose=args.verbose
        )

    # Take random subset of training data
    train_data_subset, train_labels_subset = sample_subset(
        train_data_array, train_labels_array, args.sample_size,
        train_data_rows, train_data_cols)

    if args.verbose > 0:
        print('\nTraining classifier...\n')

    # Train the classifier
    classifier = get_classifier(args.classifier, args.jobs, args.verbose)
    classifier.fit(train_data_subset, train_labels_subset)

    if args.verbose > 0:
        print('\nClassifying training data...\n')

    # Make a prediction for the training set
    train_prediction = classifier.predict(train_data_array)

    # Save training predictions for debugging purposes
    if args.debug:
        write_geotiff(
            train_prediction,
            train_labels_geo_transform,
            train_labels_projection,
            train_labels_ctable,
            train_labels_rows,
            train_labels_cols,
            args.train_labels,
            suffixes=['predictions', args.classifier],
            verbose=args.verbose
        )

    if args.verbose > 0:
        print('\nClassifying testing data...\n')

    # Make a prediction for the testing set
    test_prediction = classifier.predict(test_data_array)

    # Save testing predictions
    write_geotiff(
        test_prediction,
        test_data_geo_transform,
        test_data_projection,
        train_labels_ctable,
        test_data_rows,
        test_data_cols,
        args.test_labels,
        suffixes=['predictions', args.classifier],
        verbose=args.verbose
    )
