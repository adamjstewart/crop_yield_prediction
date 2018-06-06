"""Functions for initializing the regressor."""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def get_regressor(model, alpha=1, c=1, epsilon=0.1, verbose=0, jobs=-1):
    """Initializes a new regressor.

    Parameters:
        model (str): the regression model
        alpha (float): the regularization strength
        c (float): SVR penalty parameter
        epsilon (float): SVR epsilon
        verbose (int): the verbosity level
        jobs (int): the number of jobs to run in parallel

    Returns:
        regressor: the regressor
    """
    if model == 'linear':
        return LinearRegression(
            fit_intercept=False, normalize=False, copy_X=False, n_jobs=jobs)
    elif model == 'ridge':
        return Ridge(alpha=alpha)
    elif model == 'svr':
        return SVR(C=c, epsilon=epsilon, verbose=verbose)
    elif model == 'random-forest':
        return RandomForestRegressor(n_jobs=jobs, verbose=verbose)
    else:
        msg = "Unsupported regression model: '{}'"
        raise ValueError(msg.format(model))
