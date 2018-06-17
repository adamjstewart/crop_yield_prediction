"""Various metrics for evaluating the performance of the model."""

import collections
import colorama
import math
import sklearn
import statistics


# Conversion rates (from http://www.sagis.org.za/conversion_table.html)
BSH_AC_TO_T_HA = 0.06277

# Metrics
RMSE = 'RMSE'
R2 = 'R2 (r * r)'
R2_CLASSIC = 'R2 (classic)'

METRICS = (RMSE, R2, R2_CLASSIC)

# Metrics to beat (from Yan Li's paper)
# BASELINE = {
#     'median': {
#         RMSE: 0.986,
#         R2: 0.790,
#         R2_CLASSIC: 0.749,
#     },
#     'mean': {
#         RMSE: 1.008,
#         R2: 0.791,
#         R2_CLASSIC: 0.757,
#     },
#     'combined': {
#         RMSE: 1.027,
#         R2: 0.823,
#         R2_CLASSIC: 0.823,  # ?
#     }
# }

# Metrics to beat (from linear regression baseline)
BASELINE = {
    'median': {
        RMSE: 1.007,
        R2: 0.765,
        R2_CLASSIC: 0.722,
    },
    'mean': {
        RMSE: 1.056,
        R2: 0.746,
        R2_CLASSIC: 0.701,
    },
    'combined': {
        RMSE: 1.092,
        R2: 0.751,
        R2_CLASSIC: 0.750,
    }
}


def calculate_statistics(labels, predictions):
    """Calculates the RMSE, R2 (r * r), and R2 (classic) performance metrics.

    Parameters:
        labels (pandas.Series): the ground truth labels
        predictions (pandas.Series): the predicted labels

    Returns:
        dict: a dictionary of performance metrics
    """
    # Evaluate the performance
    mse = sklearn.metrics.mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    r = labels.corr(predictions)
    r2 = r ** 2
    r2_classic = sklearn.metrics.r2_score(labels, predictions)

    # Convert from bsh/ac to t/ha
    rmse *= BSH_AC_TO_T_HA

    stats = {
        RMSE: rmse,
        R2: r2,
        R2_CLASSIC: r2_classic,
    }

    return stats


def print_statistics(stats, type='median'):
    """Prints the RMSE, R2 (r * r), and R2 (classic) performance metrics.

    Parameters:
        rmse (float): RMSE
        r2 (float): R2 (r * r)
        r2_classic (float): R2 (classic)
        type (str): valid values: ['median', 'mean', 'combined']
    """
    string = '{:6.3f}'

    for metric in METRICS:
        # Convert to a formatted string
        result = string.format(stats[metric])

        # Compare the result against the baseline
        if metric == 'RMSE':
            if stats[metric] > BASELINE[type][metric]:
                result = colorama.Fore.RED + result
        else:
            if stats[metric] < BASELINE[type][metric]:
                result = colorama.Fore.RED + result

        # Print the result
        print(colorama.Fore.CYAN + '    {:13}'.format(metric + ':'), result)


def calculate_overall_statistics(yearly_stats):
    """Calculates the median and mean statistics over all years.

    Parameters:
        yearly_stats (dict): the yearly statistics for every year

    Returns:
        dict: a dictionary of overall statistics
    """
    nested_dict = lambda: collections.defaultdict(nested_dict)
    overall_stats = nested_dict()

    for dataset in ('train', 'test'):
        for metric in METRICS:
            results = []

            for year in yearly_stats:
                results.append(yearly_stats[year][dataset][metric])

            overall_stats[dataset]['median'][metric] = \
                statistics.median(results)
            overall_stats[dataset]['mean'][metric] = \
                statistics.mean(results)

    return overall_stats


def print_overall_statistics(overall_stats, verbose=3):
    """Prints the median, mean, and combined overall statistics.

    Parameters:
        overall_stats (dict): the overall statistics
        verbose (int): the verbosity level
    """
    if verbose > 0:
        print(colorama.Fore.GREEN + '\nYear:', 'All years')

    if verbose > 2:
        print(colorama.Fore.BLUE + '\nTraining...')

        print(colorama.Fore.MAGENTA + '\nMedian Performance:\n')
        print_statistics(overall_stats['train']['median'], type='median')
        print(colorama.Fore.MAGENTA + '\nMean Performance:\n')
        print_statistics(overall_stats['train']['mean'], type='mean')

    if verbose > 1:
        print(colorama.Fore.BLUE + '\nTesting...')

        print(colorama.Fore.MAGENTA + '\nMedian Performance:\n')
        print_statistics(overall_stats['test']['median'], type='median')
        print(colorama.Fore.MAGENTA + '\nMean Performance:\n')
        print_statistics(overall_stats['test']['mean'], type='mean')
        print(colorama.Fore.MAGENTA + '\nCombined Performance:\n')
        print_statistics(overall_stats['test']['combined'], type='combined')
