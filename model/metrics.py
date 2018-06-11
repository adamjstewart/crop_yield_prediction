"""Various metrics for evaluating the performance of the model."""

import colorama
import math
import sklearn


# Conversion metrics
BSH_AC_TO_T_HA = 0.06277

# Metrics to beat (from Yan Li's paper)
BEST_RMSE = {
    'combined': 1.027,
    'median': 0.986,
    'mean': 1.008
}

BEST_R2 = {
    'combined': 0.823,
    'median': 0.790,
    'mean': 0.791
}

BEST_R2_CLASSIC = {
    'combined': 0.823,  # ?
    'median': 0.749,
    'mean': 0.757
}


def bushels_per_acre_to_tons_per_hectare(series):
    """Converts from bsh/ac to t/ha.

    Conversion rate comes from http://www.sagis.org.za/conversion_table.html

    Parameters:
        series (pandas.Series): the values to convert in bsh/ac

    Returns:
        pandas.Series: the converted values in t/ha
    """
    return series * BSH_AC_TO_T_HA


def calculate_statistics(labels, predictions):
    """Calculates the RMSE, R2 (r * r), and R2 (classic) performance metrics.

    Parameters:
        labels (pandas.Series): the ground truth labels
        predictions (pandas.Series): the predicted labels

    Returns:
        float: RMSE
        float: R2 (r * r)
        float: R2 (classic)
    """
    # Convert from bsh/ac to t/ha
    labels = bushels_per_acre_to_tons_per_hectare(labels)
    predictions = bushels_per_acre_to_tons_per_hectare(predictions)

    # Evaluate the performance
    mse = sklearn.metrics.mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    r = labels.corr(predictions)
    r2 = r ** 2
    r2_classic = sklearn.metrics.r2_score(labels, predictions)

    return rmse, r2, r2_classic


def print_statistics(rmse, r2, r2_classic, type='combined'):
    """Prints the RMSE, R2 (r * r), and R2 (classic) performance metrics.

    Parameters:
        rmse (float): RMSE
        r2 (float): R2 (r * r)
        r2_classic (float): R2 (classic)
        type (str): valid values: ['combined', 'median', 'mean']
    """
    # Convert to a formatted string
    string = '{:6.3f}'
    rmse_str = string.format(rmse)
    r2_str = string.format(r2)
    r2_classic_str = string.format(r2_classic)

    # Compare the results against Yan Li's results
    if rmse > BEST_RMSE[type]:
        rmse_str = colorama.Fore.RED + rmse_str
    if r2 < BEST_R2[type]:
        r2_str = colorama.Fore.RED + r2_str
    if r2_classic < BEST_R2_CLASSIC[type]:
        r2_classic_str = colorama.Fore.RED + r2_classic_str

    # Print the results
    print(colorama.Fore.CYAN + '    RMSE:        ', rmse_str)
    print(colorama.Fore.CYAN + '    R2 (r * r):  ', r2_str)
    print(colorama.Fore.CYAN + '    R2 (classic):', r2_classic_str)
