"""Various metrics for evaluating the performance of the model."""


def rmse(labels, predictions):
    """Calculates the Root-Mean-Squared Error (RMSE) between
    labels and predictions.

    https://en.wikipedia.org/wiki/Root-mean-square_deviation#Formula

    Parameters:
        labels (pandas.Series): the ground truth labels
        predictions (pandas.Series): the predicted labels

    Returns:
        float: the RMSE between labels and predictions
    """
    error = labels - predictions
    squared_error = error ** 2
    mse = squared_error.mean()
    rmse = mse ** 0.5

    return rmse


def r2(labels, predictions):
    """Calculates the squared correlation coefficient (r^2) between
    labels and predictions.

    https://en.wikipedia.org/wiki/Coefficient_of_determination#As_squared_correlation_coefficient

    Parameters:
        labels (pandas.Series): the ground truth labels
        predictions (pandas.Series): the predicted labels

    Returns:
        float: the squared correlation between labels and predictions
    """
    r = labels.corr(predictions)
    r2 = r ** 2

    return r2


def r2_classic(labels, predictions):
    """Calculates the coefficient of determination (r^2) between
    labels and predictions.

    https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions

    Parameters:
        labels (pandas.Series): the ground truth labels
        predictions (pandas.Series): the predicted labels

    Returns:
        float: the coefficient of determination between labels and predictions
    """
    # Mean of the observed data
    y_bar = labels.mean()

    # Total sum of squares
    ss_tot = ((labels - y_bar) ** 2).sum()

    # Residual sum of squares
    ss_res = ((labels - predictions) ** 2).sum()

    # Coefficient of determination
    r2 = 1 - ss_res / ss_tot

    return r2
